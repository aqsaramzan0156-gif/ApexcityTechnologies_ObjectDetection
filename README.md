# 🎯 Object Detection and Tracking with YOLO

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

A real-time object detection and tracking system built with YOLO and OpenCV for the Apexcity Technologies AI Internship (Task 4).

## 📸 Demo

![Demo](demo.gif) *(Add a GIF or screenshot of your output here)*

## ✨ Features

- **Real-time Object Detection** using YOLOv3
- **Object Tracking** with unique IDs across frames
- **Supports 80+ Object Classes** from the COCO dataset
- **Processes Video Files** and displays output with bounding boxes
- **FPS Counter** to monitor performance

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- A video file for testing

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aqsaramzan0156-gif/ApexcityTechnologies_ObjectDetection.git
cd ApexcityTechnologies_ObjectDetection

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy matplotlib

# Download the YOLO weights file (237 MB)
# Link: https://pjreddie.com/media/files/yolov3.weights
# Place it in the project folder.

# Run the detection script
python main.py
```

## 🚀 How It Works

1. **Loads YOLO model** using pre-trained weights and configuration.
2. **Reads video** frame by frame.
3. **Detects objects** in each frame (people, cars, etc.).
4. **Draws bounding boxes** with class labels and confidence scores.
5. **Assigns tracking IDs** to objects for consistent identification.
6. **Displays output** in a window and saves it as `output_video.mp4`.
7. Press `q` to quit the video window.

## 📁 Project Structure

```
ApexcityTechnologies_ObjectDetection/
├── main.py              # Main Python script
├── coco.names           # COCO class labels
├── yolov3.cfg           # YOLO configuration file
├── yolov3.weights       # YOLO weights (download separately)
├── test_video.mp4       # Input video (not included)
├── output_video.mp4     # Output video (generated)
├── README.md            # This file
└── .gitignore           # Ignored files
```

## 📋 Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib

## 📄 License

MIT License — Free to use, modify, and distribute.

---

<p align="center">
  <sub>Built with ❤️ in Python for ApexcityTechnologies AI Internship — Task 4</sub>
</p> 