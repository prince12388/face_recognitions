# Face Recognition System

A **Face Recognition System** built using **Keras (Deep Learning)**, **Haarcascade (Classical CV)**, and **Machine Learning algorithms**.  
This project demonstrates the end-to-end pipeline of **face detection, feature extraction, training, and recognition**, designed with industry-level coding practices and scalability in mind.  

---

## 🚀 Features  
- **Face Detection**: Uses OpenCV Haarcascade classifier for fast and efficient face detection.  
- **Deep Learning Model (Keras)**: Pre-trained/custom CNN model for robust feature extraction.  
- **ML Algorithms**: Classification and verification using algorithms such as **SVM, KNN, or Logistic Regression**.  
- **Real-time Inference**: Supports live video stream recognition via OpenCV.  
- **Modular Codebase**: Separation of concerns for **data preprocessing, model training, and evaluation**.  
- **Extendable**: Can integrate with **REST APIs, microservices, or cloud deployment (AWS/GCP/Azure)**.  

---

## 📂 Project Structure  

```
face-recognition/
│
├── data/                   # Dataset (images, embeddings)
│   ├── raw/                # Raw face images
│   └── processed/          # Preprocessed & aligned faces
│
├── models/                 # Trained models
│   ├── keras_model.h5      # CNN-based feature extractor
│   ├── classifier.pkl      # Trained ML classifier
│   └── haarcascade_frontalface.xml  # Haarcascade detector
│
├── notebooks/              # Jupyter notebooks for experiments
│
├── src/                    # Source code
│   ├── preprocessing.py    # Face alignment & preprocessing
│   ├── feature_extractor.py# CNN feature extraction
│   ├── train.py            # Model training pipeline
│   ├── recognize.py        # Real-time face recognition
│   └── utils.py            # Helper functions
│
├── requirements.txt        # Dependencies
├── README.md               # Documentation
└── LICENSE                 # License
```

---

## ⚙️ Installation  

### Prerequisites  
- Python **3.8+**  
- Virtual environment (recommended)  

### Setup  
```bash
git clone https://github.com/your-username/face-recognition.git
cd face-recognition

# Create virtual env
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧑‍💻 Usage  

### 1. **Prepare Dataset**  
Organize dataset in the following format:  
```
data/raw/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── person2/
│   ├── img1.jpg
│   ├── img2.jpg
```

### 2. **Train Model**  
```bash
python src/train.py --dataset data/raw --output models/
```

### 3. **Run Real-time Recognition**  
```bash
python src/recognize.py --model models/keras_model.h5 --classifier models/classifier.pkl
```

---

## 📊 Pipeline  

1. **Detection**: Haarcascade detects face regions.  
2. **Preprocessing**: Resize, normalize, and align faces.  
3. **Feature Extraction**: CNN (Keras) extracts embeddings.  
4. **Classification**: ML model (e.g., SVM/KNN) predicts identity.  
5. **Recognition**: Output recognized person or "Unknown".  

---

## 🔬 Technologies Used  

- **Languages**: Python, NumPy, Pandas  
- **Deep Learning**: TensorFlow / Keras  
- **Computer Vision**: OpenCV (Haarcascade)  
- **ML Algorithms**: scikit-learn (SVM, KNN, Logistic Regression)  
- **Utilities**: joblib, matplotlib, argparse  

---

## 📈 Performance  

- **Detection Accuracy**: ~95% (frontal faces, good lighting)  
- **Recognition Accuracy**: ~90%+ (custom dataset with >500 images/person)  
- Optimized for **low-latency real-time applications**.  

---

## 🏗️ Scalability & Improvements  

- Replace Haarcascade with **MTCNN / Dlib / RetinaFace** for better detection.  
- Use **FaceNet / ArcFace embeddings** for more robust recognition.  
- Deploy as **REST API (FastAPI/Flask)** or integrate with **Docker + Kubernetes**.  
- Cloud hosting on **AWS (SageMaker, EC2, S3)**.  

---

## 📜 License  

This project is licensed under the **MIT License**.  

---

## 🤝 Contributing  

Contributions are welcome!  
- Fork the repo  
- Create a feature branch (`git checkout -b feature-name`)  
- Commit changes  
- Open a Pull Request  

---

## 👨‍💻 Author  

**Prince Yadav** – Software Development Engineer | Machine Learning Enthusiast  

