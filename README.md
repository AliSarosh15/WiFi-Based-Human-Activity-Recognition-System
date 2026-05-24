# 📡 WiFi-Based Human Activity Recognition System

A Machine Learning-based Human Activity Recognition (HAR) system that uses WiFi RSSI signals and ESP32 devices to detect and classify human activities without requiring cameras or wearable sensors.

This project explores how human movement affects WiFi signal propagation and uses those variations for real-time activity recognition using Machine Learning techniques.

---

# 🚀 Features

- 📶 WiFi RSSI-based activity sensing
- 🤖 Machine Learning powered classification
- ⚡ Real-time activity prediction
- 📊 Signal processing & feature extraction
- 🧠 Random Forest-based HAR pipeline
- 📡 ESP32 + WiFi Router setup
- 🔬 Research-oriented implementation
- 💰 Low-cost hardware solution
- 📈 Domain evaluation experiments
- 🧪 CSI vs RSSI comparison

---

# 🧠 Activities Recognized

- Walking
- Standing
- Sitting
- Lying
- Motion Detection

---

# 🏗️ System Workflow

```text
WiFi Router
     ↓
ESP32 captures RSSI fluctuations
     ↓
Serial Communication to Python
     ↓
Feature Extraction Pipeline
     ↓
Machine Learning Model
     ↓
Real-Time Activity Prediction
```

---

# 🛠️ Tech Stack

## Languages & ML
- Python
- Scikit-learn
- NumPy
- Pandas
- SciPy

## Hardware
- ESP32
- WiFi Router

## Visualization & Live Demo
- OpenCV
- PySerial

## Development Tools
- Arduino IDE
- Jupyter Notebook

---

# 📂 Project Structure

```text
WiFi-Based-Human-Activity-Recognition-System/
│
├── Data/
├── changed_domain/
├── csi comparision/
├── live_har/
├── same_domain/
├── .gitignore
└── README.md
```

---

# 📖 Folder Description

| Folder | Description |
|---|---|
| `Data/` | Contains collected RSSI datasets |
| `same_domain/` | Same environment experiments |
| `changed_domain/` | Cross-domain evaluation |
| `csi comparision/` | CSI vs RSSI comparison |
| `live_har/` | Real-time HAR implementation |

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/AliSarosh15/WiFi-Based-Human-Activity-Recognition-System.git

cd WiFi-Based-Human-Activity-Recognition-System
```

## 2️⃣ Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib opencv-python pyserial
```

---

# ▶️ Running the Project

## 🔹 Same Domain Evaluation

```bash
cd same_domain
python part1.py
```

## 🔹 Changed Domain Evaluation

```bash
cd changed_domain
python part2.py
```

## 🔹 CSI Comparison

```bash
cd "csi comparision"
python part3.py
```

## 🔹 Real-Time HAR Demo

```bash
cd live_har
python live_har.py
```

---

# 📊 Results

| Experiment | Accuracy |
|---|---|
| Same Domain Testing | ~80% |
| Cross Domain Testing | ~33% |
| Binary Motion Detection | ~81% |

---

# 🤖 Model Used

```python
RandomForestClassifier(
    n_estimators=500,
    random_state=42
)
```

---

# 💡 Future Improvements

- Deep Learning models (CNN/LSTM)
- Multi-person activity detection
- Domain adaptation techniques
- Edge AI deployment
- Gesture recognition

---

# 📡 Hardware Requirements

| Hardware | Purpose |
|---|---|
| ESP32 Dev Board | RSSI data collection |
| WiFi Router | Signal transmitter |
| Laptop/Desktop | ML processing |
| USB Cable | Serial communication |

---

# 🤝 Contributing

Contributions are welcome.

```bash
Fork → Clone → Create Branch → Commit → Push → Pull Request
```

---

# 👨‍💻 Author

## Ali Sarosh

🎓 BTech CSE Student  
💻 Backend Developer & AI/ML Enthusiast  
🚀 Open Source Contributor

### Interests
- Backend Development
- Flask & FastAPI
- Machine Learning
- WiFi Sensing Research
- Open Source Development

### Links
- GitHub: https://github.com/AliSarosh15
- LinkedIn: http://www.linkedin.com/in/ali-sarosh-332b90280/

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🧠 Share feedback & ideas

---

# 📷 Project Goal

This project demonstrates how low-cost wireless sensing systems can be used for intelligent activity recognition applications in:

- Smart Homes
- Healthcare Monitoring
- Elderly Care
- Security Systems
- Ambient Intelligence Systems
