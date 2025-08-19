# 🏋️ AI Trainer with Mediapipe

This project implements an AI-powered fitness trainer using Python, OpenCV, and Mediapipe.  
It uses pose detection to count exercise repetitions (like bicep curls) in real time and provides visual feedback.

---

## 🚀 Features
- Real-time human pose detection with Mediapipe.
- Counts repetitions based on joint angles.
- Displays percentage progress bar and rep count.
- Smooth performance with optimized OpenCV rendering.
- Works with webcam input.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/faisal-ajao/ai-trainer-mediapipe.git
cd ai-trainer-mediapipe

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the main script:
Run:
```
python main.py
```
### Notes:
- The system will open your webcam and start detecting your movements.
- Reps are counted automatically when you complete full motion cycles.
- Progress percentage and count are displayed live.

---

## 📊 Output Example (Video)
[![Watch the output](https://img.youtube.com/vi/LrVoDBY68iU/hqdefault.jpg)](https://youtu.be/LrVoDBY68iU?feature=shared)

---

## 📂 Project Structure
```
ai-trainer-mediapipe/
├── PoseModule.py          # Custom pose detection module using Mediapipe
├── main.py                # Main AI trainer script
├── README.md
└── requirements.txt
```

---

## 🧠 Tech Stack
- Python 3.11.5
- OpenCV
- Mediapipe
- NumPy

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Install dependencies
```bash
pip install -r requirements.txt
```
