import os
import re
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from ultralytics import YOLO

from paddleocr import PaddleOCR

import firebase_admin
from firebase_admin import credentials, storage
import smtplib
from email.mime.text import MIMEText
import google.generativeai as genai
app = Flask(__name__)

# Configure Flask app
app.config["SECRET_KEY"] = "Hi@123"
app.config["MONGO_URI"] = "mongodb+srv://kirantummala36:Kiran6104@cluster0.gxehi.mongodb.net/Helmet?retryWrites=true&w=majority&appName=Cluster0"
app.config["JWT_SECRET_KEY"] = "Hi@123"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
GOOGLE_API_KEY = 'MY-API-KEY'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')
# Folders
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
CROPPED_FOLDER = "static/cropped"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["CROPPED_FOLDER"] = CROPPED_FOLDER

# Initialize PyMongo
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Initialize Firebase Admin SDK with Storage
cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'idx-8b7b5.appspot.com'  # Replace with your Firebase Storage bucket
})

# Initialize Firebase Storage
bucket = storage.bucket()

# Load YOLO models
coco_yolo = YOLO("yolov8n.pt")  # for person and motorcycle   pretrained model
helmet_yolo = YOLO("models/best.pt")  # your trained helmet detection model        custom model
license_plate_yolo = YOLO("models/LicensePlateDetection.pt")  # license plate model    custom model

# Initialize PaddleOCR
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")
mode = False

# IoU Function
def iou(box1, box2):
    x1, y1, x2, y2 = [int(v) for v in box1]
    x1b, y1b, x2b, y2b = [int(v) for v in box2]
    inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
    inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

@app.route('/chat', methods=['POST'])
def chat_response():
    data = request.get_json()
    user_input = data.get('message')

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = model.generate_content(user_input).text
        clean_response = response.replace('*', '').strip()
        return jsonify({"response": clean_response})

    except Exception as e:
        print(f"Error from Gemini API: {e}")
        return jsonify({"error": "Error communicating with chatbot."}), 500
    
# License Plate OCR
def extract_license_plate(region):
    plate_results = license_plate_yolo(region)
    boxes = plate_results[0].boxes.data.cpu().numpy() if plate_results[0].boxes is not None else []
    for d in boxes:
        if int(d[5]) == 3:  # Assuming class 3 is license plate
            x1, y1, x2, y2 = map(int, d[:4])
            plate_crop = region[y1:y2, x1:x2]
            cv2.rectangle(region, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(region, "Plate", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            result = ocr_model.ocr(plate_crop, cls=True)
            if result and result[0]:
                return " ".join([r[1][0] for r in result[0]])
    return "Number Plate Not Found"

# Function to send email notification
def send_fine_notification(email, license_plate, fine_amount, date_issued):
    sender_email = "trafficchallan12@gmail.com"  # Replace with your Gmail address
    sender_password = "oitqjmsuvudcraly"  # Replace with your App Password
    receiver_email = email  # Use the user's email from the database

    subject = "Traffic Violation Fine Notification"
    body = f"""
    Dear User,

    A traffic violation has been detected associated with your license plate: {license_plate}.

    Fine Details:
    - Fine Amount: Rs.{fine_amount}
    - Date Issued: {date_issued}
    - Status: Unpaid

    Please settle the fine at your earliest convenience through the dashboard.

    Regards,
    Traffic Monitoring System
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"Email sent successfully to {receiver_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Frame-wise processing
def process_frame(image):
    data = []
    result = coco_yolo(image)[0]
    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else []

    # Identify all motorcycles and persons
    motorcycle_boxes = [b[:4] for b in boxes if int(b[5]) == 3]
    person_boxes = [b[:4] for b in boxes if int(b[5]) == 0]

    for mbox in motorcycle_boxes:
        x1, y1, x2, y2 = map(int, mbox)
        related_persons = [p for p in person_boxes if iou(mbox, p) > 0.1]
        if not related_persons:
            continue

        # Expand bounding box to include all related persons
        for p in related_persons:
            x1 = min(x1, int(p[0]))
            y1 = min(y1, int(p[1]))
            x2 = max(x2, int(p[2]))
            y2 = max(x2, int(p[3]))

        region = image[y1:y2, x1:x2]

        # Helmet detection in region
        helmet_result = helmet_yolo(region)[0]
        helmet_data = helmet_result.boxes.data.cpu().numpy() if helmet_result.boxes else []
        
        # Filter only helmet-related classes (skip rider class 2)
        helmet_classes = [(int(h[5]), h[:4]) for h in helmet_data if int(h[5]) in [0, 1]]
        helmet_boxes = [hbox for cls, hbox in helmet_classes]
        helmet_class_ids = [cls for cls, _ in helmet_classes]

        # Draw helmet boxes
        for cls_id, hb in helmet_classes:
            hx1, hy1, hx2, hy2 = map(int, hb)
            label = "With Helmet" if cls_id == 0 else "Without Helmet"
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
            cv2.rectangle(region, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(region, label, (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Determine helmet status
        helmet_status = "Wearing Helmet"
        if 1 in helmet_class_ids:  # 1 = without helmet
            helmet_status = "No Helmet Detected"

        # License plate extraction only if violation
        plate_text = "Not Detected"
        if helmet_status == "No Helmet Detected":
            plate_text = extract_license_plate(region)

        # Save cropped region
        crop_name = f"cropped_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
        crop_path = os.path.join(CROPPED_FOLDER, crop_name)
        cv2.imwrite(crop_path, region)

        data.append((crop_path, plate_text, len(related_persons), helmet_status))

        # Draw bounding box on original image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return data, image

# Main Route
@app.route("/")
def index():
    return redirect(url_for("login_page"))

@app.route("/admin/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Clear cropped folder
        for f in os.listdir(app.config['CROPPED_FOLDER']):
            os.remove(os.path.join(app.config['CROPPED_FOLDER'], f))

        file = request.files["file"]
        if not file or file.filename == "":
            return redirect(request.url)

        # Generate a clean filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize the original filename to remove invalid characters
        original_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_', '-'))
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the file with error handling
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        is_video = filename.lower().endswith((".mp4", ".avi", ".mov"))
        cropped_data = []
        preview_frame = None
        fine_messages = []  # To store messages about issued fines

        if is_video:
            mode = False
            cap = cv2.VideoCapture(filepath)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 10 != 0:
                    frame_idx += 1
                    continue
                result_data, processed_frame = process_frame(frame)
                cropped_data.extend(result_data)
                if preview_frame is None:
                    preview_frame = processed_frame.copy()
                frame_idx += 1
            cap.release()
            processed_filename = f"processed_{filename.split('.')[0]}.jpg"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, preview_frame)
        else:
            mode = True
            image = cv2.imread(filepath)
            cropped_data, processed_img = process_frame(image)
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, processed_img)
            preview_frame = processed_img

        # Process cropped_data to issue fines for violations
        for data in cropped_data:
            path, license_plate, persons, helmet = data
            if helmet == "No Helmet Detected" and license_plate != "-" and license_plate != "Number Plate Not Found":
                # Find user associated with the license plate
                user = mongo.db.users.find_one({"license_plate": license_plate})
                if user:
                    # Create fine record (no duplicate check)
                    fine_record = {
                        "_id": ObjectId(),
                        "license_plate": license_plate,
                        "fine_amount": 500,
                        "status": "Unpaid",
                        "date_issued": datetime.now().strftime("%Y-%m-%d"),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Added for tracking exact time
                    }
                    result = mongo.db.challans.insert_one(fine_record)
                    # Send email notification
                    send_fine_notification(user["email"], license_plate, fine_record["fine_amount"], fine_record["date_issued"])
                    fine_messages.append(f"Fine issued for license plate {license_plate} at {fine_record['timestamp']}")
                else:
                    fine_messages.append(f"User with license plate {license_plate} not found")

        # Set message based on processing results
        if mode:
            try:
                if not cropped_data:
                    message = "No data found in processed image. Please use another image."
                elif cropped_data[0][3] == "Wearing Helmet":
                    cropped_data[0] = (cropped_data[0][0], "-", cropped_data[0][2], cropped_data[0][3])
                    message = "No Violation Found"
                else:
                    message = "; ".join(fine_messages) if fine_messages else "Violations processed"
            except (IndexError, TypeError) as e:
                message = f"Error processing image data: {str(e)}"
        else:
            message = "; ".join(fine_messages) if fine_messages else "No Violation Found"

        return render_template(
            "admin/results.html",
            uploaded_image=url_for('static', filename=f'uploads/{filename}'),
            processed_image=url_for('static', filename=f'processed/{processed_filename}'),
            cropped_data=[
                (
                    url_for('static', filename=f'cropped/{os.path.basename(path)}'),
                    text,
                    persons,
                    helmet
                ) for path, text, persons, helmet in cropped_data
            ],
            message=message
        )
    return render_template("admin/index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "POST":
        data = request.form
        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        password = data.get("password")
        confirm_password = data.get("confirm_password")
        license_plate = data.get("license_plate")

        if password != confirm_password:
            return redirect(url_for("signup_page", error="Passwords do not match"))

        if mongo.db.users.find_one({"email": email}):
            return redirect(url_for("signup_page", error="Email already exists"))

        file = request.files.get("license_image")
        if not file or file.filename == "":
            return redirect(url_for("signup_page", error="No selected file"))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

        user = {
            "name": name,
            "email": email,
            "phone": phone,
            "password": hashed_pw,
            "license_plate": license_plate,
            "license_image": file_path
        }

        mongo.db.users.insert_one(user)
        return redirect(url_for("login_page"))

    error = request.args.get("error")
    return render_template("signup.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        data = request.form
        email = data.get("email")
        password = data.get("password")
        
        user = mongo.db.users.find_one({"email": email})
        if not user or not bcrypt.check_password_hash(user["password"], password):
            return redirect(url_for("login_page", error="Invalid email or password"))
        
        session["user_email"] = email
        return redirect(url_for("dashboard"))

    error = request.args.get("error")
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("login_page"))

# Function to convert Firebase timestamp
def parse_relative_time(timestamp_str):
    try:
        parts = timestamp_str.split(", ")
        total_seconds = 0

        for part in parts:
            value, unit = part.split(" ")
            value = int(value)
            if "hour" in unit:
                total_seconds += value * 3600
            elif "minute" in unit:
                total_seconds += value * 60
            elif "second" in unit:
                total_seconds += value

        minutes_ago = total_seconds // 60
        return f"{minutes_ago} minutes ago"
    except (ValueError, AttributeError):
        return timestamp_str or "N/A"

@app.route('/admin/cam')
def display_images():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    # List files in Firebase Storage (e.g., in the 'images/' directory)
    blobs = bucket.list_blobs(prefix='images/')  # Adjust prefix as needed
    images = []

    for blob in blobs:
        # Generate a signed URL for the image (expires in 1 hour)
        url = blob.generate_signed_url(
            expiration=timedelta(hours=1),
            method='GET'
        )
        # Get metadata (e.g., timestamp)
        metadata = blob.metadata or {}
        timestamp = metadata.get('timestamp', blob.time_created.strftime('%Y-%m-%d %H:%M:%S')) if blob.metadata else blob.time_created.strftime('%Y-%m-%d %H:%M:%S')

        images.append({
            'id': blob.name,
            'url': url,
            'timestamp': timestamp
        })

    return render_template('admin/camera.html', images=images)

@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return redirect(url_for("login_page", error="Please log in first"))

    user = mongo.db.users.find_one({"email": session["user_email"]})
    
    if not user:
        return redirect(url_for("login_page", error="User not found"))

    return render_template("dashboard.html", user=user)

@app.route("/profile")
def profile():
    if "user_email" not in session:
        return redirect(url_for("login_page", error="Please log in first"))

    user = mongo.db.users.find_one({"email": session["user_email"]})
    
    if not user:
        return redirect(url_for("login_page", error="User not found"))

    return render_template("profile.html", user=user)

@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():
    if "user_email" not in session:
        return redirect(url_for("login_page"))

    user = mongo.db.users.find_one({"email": session["user_email"]})

    if request.method == "POST":
        name = request.form.get("name")
        phone = request.form.get("phone")
        license_plate = request.form.get("license_plate")
        license_image = request.files.get("license_image")

        update_data = {
            "name": name,
            "phone": phone,
            "license_plate": license_plate
        }

        if license_image and license_image.filename:
            filename = secure_filename(license_image.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            license_image.save(filepath)
            update_data["license_image"] = filepath

        mongo.db.users.update_one({"email": session["user_email"]}, {"$set": update_data})
        return redirect(url_for("profile"))

    return render_template("edit_profile.html", user=user)

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email == "admin@gmail.com" and password == "admin123":
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("admin_login", error="Invalid credentials"))

    error = request.args.get("error")
    return render_template("admin/login.html", error=error)

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    challans = list(mongo.db.challans.find())
    users_data = {user["license_plate"]: user for user in mongo.db.users.find({}, {"_id": 0, "password": 0})}

    for challan in challans:
        license_plate = challan["license_plate"]
        challan["user_details"] = users_data.get(license_plate, None)

        if "date_issued" in challan and isinstance(challan["date_issued"], str):
            try:
                challan["date_issued"] = datetime.strptime(challan["date_issued"], "%Y-%m-%d")
            except ValueError:
                challan["date_issued"] = None

    total_challans = len(challans)
    pending_fines = sum(challan["fine_amount"] for challan in challans if challan["status"] == "Unpaid")
    collected_revenue = sum(challan["fine_amount"] for challan in challans if challan["status"] == "Paid")

    return render_template(
        "admin/dashboard.html",
        challans=challans,
        total_challans=total_challans,
        pending_fines=pending_fines,
        collected_revenue=collected_revenue
    )

@app.route('/admin/update_challan_status/<challan_id>', methods=['POST'])
def update_challan_status(challan_id):
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized access"}), 401

    try:
        # Update the challan status in MongoDB
        result = mongo.db.challans.update_one(
            {"_id": ObjectId(challan_id)},
            {"$set": {"status": "Paid"}}
        )
        if result.modified_count > 0:
            return jsonify({"success": "Challan status updated to Paid"}), 200
        else:
            return jsonify({"error": "Challan not found or already Paid"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to update challan: {str(e)}"}), 500

@app.route("/admin/users")
def admin_users():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    users = list(mongo.db.users.find({}, {"_id": 0}))
    for user in users:
        if "fines_pending" not in user:
            user["fines_pending"] = 0
    return render_template("admin/users.html", users=users)

@app.route("/admin/challans")
def admin_challans():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    challans = list(mongo.db.challans.find())
    return render_template("admin/challans.html", challans=challans)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

if __name__ == "__main__":
    app.run(debug=True)
