# 🤟 Indian Sign Language Recognition System

A real-time Indian Sign Language (ISL) recognition system using **MediaPipe** and **Deep Learning**, capable of detecting both **single-hand and double-hand gestures** and converting them into **text and speech output**.

---

## 🚀 Features

- 🔍 Real-time hand gesture detection using webcam  
- ✋ Supports **single-hand and double-hand gestures**  
- 🧠 Deep learning models for classification  
- 🔊 Converts predictions into **speech (Text-to-Speech)**  
- 🧾 Builds continuous **text output from gestures**  
- 🧩 Modular and reusable project structure  
- 💻 Runs locally (no need for Google Colab)

---

## 🧠 Tech Stack

- Python  
- TensorFlow / Keras  
- MediaPipe  
- OpenCV  
- NumPy  
- gTTS (Google Text-to-Speech)

---

## 📁 Project Structure
## 📁 Project Structure

```
sign_double/
│
├── dataset/                  # (Not included - user must download)
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── dataset_builder.py
│   ├── models/
│   ├── train/
│   └── evaluation.py
│
├── app/
│   ├── sign_double.py       # Real-time application
│
├── outputs/
│   ├── models/
│   ├── classes/
│   └── history/
│
├── main.py                  # Training pipeline
├── requirements.txt
└── README.md
```
## 📊 Dataset

This project uses the **Indian Sign Language dataset** from Kaggle:

👉https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset

### 📌 Setup Dataset

After downloading, place it like this:
dataset/
A/
B/
C/
...

Each folder should contain images of that gesture.

---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/keerthanarajch/Indian-sign-language-recognition.git

cd Indian-sign-language-recognition

### 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate # Linux / Mac
venv\Scripts\activate # Windows


### 3️⃣ Install dependencies
pip install -r requirements.txt


---

## 🏋️ Train the Model
python3 main.py


This will:

- Build dataset from images  
- Train single-hand and double-hand models  
- Save models in `outputs/models/`  

---

## 🎥 Run Real-Time Application
python app/main_app.py


### Controls:

- Press **Q** → Quit  
- Press **C** → Clear text  

---

## 📈 Output

After training:
outputs/
models/
model_single.h5
model_double.h5

classes/
classes_single.json
classes_double.json


---

## 💡 How It Works

1. MediaPipe detects hand landmarks  
2. Landmarks are normalized  
3. Features are passed to trained models  
4. Model predicts gesture class  
5. Stable predictions are converted to text  
6. Text is converted to speech  

---

## 🎯 Applications

- Assistive technology for hearing/speech impaired  
- Real-time communication systems  
- Human-computer interaction  
- Educational tools  

---

## ⚠️ Notes

- Dataset is not included due to size and licensing  
- Models can be regenerated using `main.py`  
- Works on CPU (GPU not required)

---

## 🙌 Acknowledgements

- Google MediaPipe  
- TensorFlow  
- Kaggle dataset contributors  

---

## 👩‍💻 Author

Keerthanaraj C H  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
