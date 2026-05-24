📡 WiFi-Based Human Activity Recognition System

A Machine Learning-based Human Activity Recognition (HAR) system that uses WiFi RSSI signals and ESP32 devices to detect and classify human activities without requiring cameras or wearable sensors.

This project explores how human movement affects WiFi signal propagation and uses those variations for real-time activity recognition using Machine Learning techniques.


🚀 Features
📶 WiFi RSSI-based activity sensing
🤖 Machine Learning powered classification
⚡ Real-time activity prediction
📊 Signal processing & feature extraction
🧠 Random Forest-based HAR pipeline
📡 ESP32 + WiFi Router setup
🔬 Research-oriented implementation
💰 Low-cost hardware solution
📈 Domain evaluation experiments
🧪 CSI vs RSSI comparison
🧠 Activities Recognized
Walking
Standing
Sitting
Lying
Motion Detection
🏗️ System Workflow
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
🛠️ Tech Stack
Languages & ML
Python
Scikit-learn
NumPy
Pandas
SciPy
Hardware
ESP32
WiFi Router
Visualization & Live Demo
OpenCV
PySerial
Development Tools
Arduino IDE
Jupyter Notebook
📂 Project Structure
WiFi-Based-Human-Activity-Recognition-System/
│
├── Data/
│
├── changed_domain/
│
├── csi comparision/
│
├── live_har/
│
├── same_domain/
│
├── .gitignore
│
└── README.md
📖 Folder Description
Folder	Description
Data/	Contains collected RSSI datasets used for training and evaluation
same_domain/	Experiments where training & testing environments are similar
changed_domain/	Cross-domain experiments for testing generalization
csi comparision/	Comparison between RSSI and CSI-based HAR approaches
live_har/	Real-time Human Activity Recognition implementation
.gitignore	Git ignored files configuration
README.md	Project documentation
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/AliSarosh15/WiFi-Based-Human-Activity-Recognition-System.git

cd WiFi-Based-Human-Activity-Recognition-System
2️⃣ Install Dependencies
pip install numpy pandas scipy scikit-learn matplotlib opencv-python pyserial
▶️ Running the Project
🔹 Same Domain Evaluation
cd same_domain
python part1.py

This experiment evaluates the model when training and testing are performed in similar environments.

🔹 Changed Domain Evaluation
cd changed_domain
python part2.py

This experiment tests how well the model generalizes across different environments and sessions.

🔹 CSI Comparison
cd "csi comparision"
python part3.py

This module compares RSSI-based HAR with CSI-based sensing systems.

🔹 Real-Time HAR Demo
cd live_har
python live_har.py

Runs the real-time Human Activity Recognition system using live RSSI data from ESP32.

📊 Results
Experiment	Accuracy
Same Domain Testing	~80%
Cross Domain Testing	~33%
Binary Motion Detection	~81%
🔬 Research Insights
✅ RSSI-Based HAR

RSSI-based Human Activity Recognition provides:

Low deployment cost
Device-free sensing
Privacy-preserving monitoring
Lightweight implementation
❌ Domain Shift Challenge

Performance decreases significantly when:

Device positions change
Environmental conditions vary
Signal reflections change
Different sessions are used

This demonstrates the domain adaptation challenge in RSSI-based sensing systems.

✅ CSI-Based Systems

CSI captures more detailed channel information than RSSI and generally achieves:

Higher accuracy
Better robustness
Improved activity separation
📈 Machine Learning Pipeline

The project includes:

Signal preprocessing
Sliding window segmentation
Statistical feature extraction
Frequency-domain analysis
Random Forest classification
🧪 Features Extracted
Time Domain Features
Mean
Variance
Standard Deviation
Signal Energy
Frequency Domain Features
FFT Features
Spectral Analysis
Entropy
Signal Dynamics
Zero Crossings
RSSI Variations
Window-based statistics
🤖 Model Used
RandomForestClassifier(
    n_estimators=500,
    random_state=42
)
Why Random Forest?
Handles noisy RSSI data effectively
Works well with handcrafted features
Robust against overfitting
Good baseline for HAR systems
💡 Future Improvements
Deep Learning models (CNN/LSTM)
Multi-person activity detection
CSI-based real-time implementation
Domain adaptation techniques
Edge AI deployment
Gesture recognition
Better real-world generalization
📡 Hardware Requirements
Hardware	Purpose
ESP32 Dev Board	RSSI data collection
WiFi Router	Signal transmitter
Laptop/Desktop	ML processing
USB Cable	Serial communication
🤝 Contributing

Contributions are welcome.

Steps
Fork → Clone → Create Branch → Commit → Push → Pull Request
📜 License

This project is licensed under the MIT License.

👨‍💻 Author
Ali Sarosh

🎓 BTech CSE Student
💻 Backend Developer & AI/ML Enthusiast
🚀 Open Source Contributor

Interests
Backend Development
Flask & FastAPI
Machine Learning
WiFi Sensing Research
Open Source Development

🔗 GitHub:
https://github.com/AliSarosh15

LinkedIn:
http://www.linkedin.com/in/ali-sarosh-332b90280/

⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🧠 Share feedback & ideas
📚 References
WiFi-based Human Activity Recognition
RSSI Signal Processing
CSI-Based Wireless Sensing
Machine Learning for HAR Systems
📷 Project Goal

This project aims to demonstrate how low-cost wireless sensing systems can be used for intelligent activity recognition applications in:

Smart Homes
Healthcare Monitoring
Elderly Care
Security Systems
Ambient Intelligence Systems
