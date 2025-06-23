# Full-Stack Intelligent Traffic Monitoring , Violation Management and fine collection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Object%20Detection-YOLOv8-blueviolet)](https://github.com/ultralytics/ultralytics)
[![OCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)](https://github.com/PaddlePaddle/PaddleOCR)

An end-to-end system that automates the detection of motorcycle helmet violations from traffic imagery. It uses a YOLOv8 model to identify riders without helmets and a high-accuracy PaddleOCR engine to read their license plates for automated enforcement.

---

## 📖 Table of Contents
* [Problem Statement](#-problem-statement)
* [Our Solution](#-our-solution)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Technology Stack](#-technology-stack)
* [Methodology](#-methodology)
* [Performance & Results](#-performance--results)
* [Setup and Installation](#-setup-and-installation)
* [Future Enhancements](#-future-enhancements)


---

## 📍 Problem Statement

Non-compliance with helmet laws is a major cause of fatalities and severe injuries among motorcycle riders, especially in developing countries. Traditional enforcement methods rely on manual monitoring by traffic police, which is **labor-intensive, prone to human error, and not scalable** for covering vast road networks. This reactive approach fails to provide consistent, real-time monitoring, leading to a high number of unrecorded violations and a persistent public safety risk.

## 💡 Our Solution

This project presents an intelligent, automated system to bridge this enforcement gap. We leverage state-of-the-art deep learning and computer vision to create a fully automated pipeline that can:
1.  **Detect** motorcyclists in an image or video frame.
2.  **Classify** whether the rider is wearing a helmet.
3.  **Identify** and **read the license plate** of any non-compliant rider.
4.  **Log** the violation with visual evidence and extracted data for review and automated fining.

By automating this process, we aim to enhance road safety, reduce fatalities, and provide law enforcement agencies with a powerful, efficient, and scalable tool.

## ✨ Key Features

- **High-Accuracy Helmet Detection**: A custom-trained YOLOv8 model distinguishes between riders with and without helmets.
- **Robust License Plate Recognition**: A multi-stage process uses YOLOv8 for precise plate localization and **PaddleOCR** for accurate text extraction, outperforming alternatives like EasyOCR.
- **End-to-End Automated Pipeline**: From image input to violation logging, the system requires no manual intervention.
- **Web-Based Interface**: A simple and intuitive UI built with **Flask** allows for easy image uploads and clear visualization of results.
- **Structured Data Logging**: Violations are automatically categorized and saved with annotated images, timestamps, and recognized license plate numbers.



## 🏗️ System Architecture

The system is designed as a modular pipeline, ensuring each stage performs its task efficiently before passing the data to the next.


## 🛠️ Technology Stack

- **Backend & Web Framework**: `Python`, `Flask`
- **Object Detection**: `YOLOv8 (Ultralytics)`
- **Optical Character Recognition (OCR)**: `PaddleOCR`
- **CV & Data Libraries**: `OpenCV`, `NumPy`, `Matplotlib`
- **Development Environment**: `Google Colab`, `Jupyter Notebook`

## 🔬 Methodology

The project's workflow follows a structured sequence:

1.  **Input Acquisition**: The system accepts an image (or video frame) from the user via the Flask web interface.
2.  **Preprocessing**: The input image is resized to `640x640` and normalized to match the YOLOv8 model's expected input.
3.  **Multi-Stage Detection**:
    - **Rider & Helmet Detection**: A custom-trained YOLOv8 model identifies `persons`, `motorcycles`, and classifies helmet status (`With Helmet` or `Without Helmet`).
    - **Violation Logic**: The system checks if a detected rider has a corresponding helmet using an Intersection over Union (IoU) threshold.
    - **License Plate Detection**: If a violation is confirmed, another YOLOv8 model is run to locate the license plate on the vehicle.
4.  **OCR Extraction**: The detected license plate region is cropped and passed to the **PaddleOCR** engine, which extracts the alphanumeric text.
5.  **Output & Visualization**: The original image is annotated with bounding boxes and labels. The result, including the recognized plate number, is displayed to the user and logged for record-keeping.

## 📊 Performance & Results

The system demonstrated high performance and reliability during evaluation.

- **License Plate Detection**: The YOLOv8 model achieved an impressive **mAP@50 of 0.983**, indicating extremely accurate localization.
- **OCR Accuracy**: The **PaddleOCR** engine achieved over **95% accuracy** on well-lit, clear, and front-facing license plates.
- **Overall Performance**: The system performs robustly in diverse scenarios but faces challenges with severe motion blur, poor lighting (night-time), and heavy occlusions, which are identified as areas for future improvement.

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smart-traffic-enforcement.git
    cd smart-traffic-enforcement
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(You will need to create a `requirements.txt` file)*
    ```bash
    pip install -r requirements.txt
    ```
    A typical `requirements.txt` would include:
    ```
    flask
    ultralytics
    paddlepaddle-gpu # or paddlepaddle for CPU
    paddleocr
    opencv-python-headless
    numpy
    matplotlib
    ```

4.  **Download Model Weights**:
    - Make sure your trained YOLOv8 model weights (`.pt` files) are placed in the designated project folder.

5.  **Run the Flask application:**
    ```bash
    python app.py
    ```

6.  **Open your browser** and navigate to `http://127.0.0.1:5000`.

## 🔮 Future Enhancements

- **Real-Time Video Processing**: Adapt the pipeline to handle live video feeds from CCTV or IP cameras.
- **Night-Time and Low-Light Enhancement**: Integrate image enhancement techniques or train the model on a specialized low-light dataset to improve night-time performance.
- **Database Integration**: Connect the system to a centralized database (e.g., government vehicle records) to automatically issue challans (fines) upon violation detection.
- **Cross-Country Deployment**: Train models on datasets from different countries to adapt to various license plate formats and traffic regulations.
- **Expand Violation Detection**: Extend the system's capabilities to detect other traffic violations, such as triple riding, speeding, or mobile phone usage.
