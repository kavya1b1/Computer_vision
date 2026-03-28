# Computer_vision

# 🚗 AI Driver Monitoring System (DMS)

An advanced **Computer Vision-based Driver Monitoring System** designed to detect driver fatigue, distraction, and unsafe behaviors in real-time using AI techniques.

---

## 📌 Overview

Road accidents due to driver fatigue and distraction are a major concern worldwide. This project addresses this issue by building an intelligent system that continuously monitors driver behavior using a webcam and provides real-time alerts.

The system combines multiple computer vision techniques to ensure accurate and reliable detection.

---

## 🚀 Features

* 😴 **Drowsiness Detection** using Eye Aspect Ratio (EAR)
* 😮 **Yawning Detection** using Mouth Aspect Ratio (MAR)
* 🧍 **Head Pose Estimation** for distraction detection
* 📱 **Mobile Phone Detection** using YOLOv8
* 📊 **Fatigue Score Calculation** (multi-factor analysis)
* ⚠️ **Risk Classification** (Low / Medium / High)
* 🔊 **Voice Alert System** for high-risk situations
* 🎥 Real-time webcam-based monitoring

---

## 🧠 Tech Stack

* **Python**
* **OpenCV** – Video processing
* **MediaPipe** – Face landmark detection
* **NumPy** – Mathematical computations
* **YOLOv8 (Ultralytics)** – Object detection (phone usage)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kavya1b1/Computer_vision.git
cd Computer_vision
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv cv_env
source cv_env/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy ultralytics
```

---

## ▶️ How to Run

```bash
python main.py
```

* Press `ESC` to exit the application

---

## 📂 Project Structure

```
Driver-Monitoring-System/
│── main.py
│── README.md
│── requirements.txt
```

---

## 📊 How It Works

1. **Face Detection & Landmarks**

   * MediaPipe extracts facial key points.

2. **Eye Monitoring**

   * EAR is calculated to detect prolonged eye closure.

3. **Yawning Detection**

   * MAR is used to detect mouth opening.

4. **Head Movement**

   * Head position is analyzed to detect distraction.

5. **Mobile Detection**

   * YOLOv8 detects if a phone is being used.

6. **Fatigue Scoring**

   * Combined signals generate a fatigue score.

7. **Alert System**

   * High fatigue triggers a voice warning.

---

## 📈 Output

* Real-time display with:

  * EAR & MAR values
  * Driver status (Active / Drowsy / Distracted / Phone Usage)
  * Fatigue Score
  * Risk Level

---

## 🎯 Applications

* Smart vehicle safety systems
* Driver assistance technologies
* Fleet monitoring systems
* Road safety research

---

## ⚠️ Limitations

* Performance may vary in low lighting
* Requires clear face visibility
* YOLO detection may slightly reduce FPS on low-end systems

---

## 🔮 Future Improvements

* 📊 Dashboard with analytics (Streamlit)
* 📱 Mobile app integration
* 🧠 Deep learning-based activity recognition
* 🚗 Integration with IoT systems

---

## 🙌 Acknowledgements

* MediaPipe by Google
* Ultralytics YOLOv8
* OpenCV community

---

## 📬 Contact

**Kavya Gupta**
📧 [kavya1b1@gmail.com](mailto:kavya1b1@gmail.com)
🔗 LinkedIn: https://www.linkedin.com/in/its-kavya-gupta/

---

⭐ If you found this project useful, consider giving it a star!
