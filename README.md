
# 🧠 Thyroid Cancer Detector

A full-stack AI-based diagnostic tool for detecting thyroid cancer using deep learning (CNN) with a login/register interface built using Node.js and a Flask backend. It also includes an intelligent chatbot to answer thyroid-related health queries.

---

## 📌 Key Features

- 🧪 **CNN-based Classification**: Detects Benign or Malignant thyroid cancer from ultrasound images.
- 🤖 **AI Chatbot**: Uses sentence embeddings (SentenceTransformers) to respond to medical queries.
- 🔐 **Login/Register Interface**: Built using Node.js/Express (frontend).
- 📊 **Model Evaluation**: Precision, Recall, F1-score, Accuracy calculation.
- 🧠 **Training Scripts**: For both classification (CNN) and segmentation (U-Net).
- 🧾 **REST API**: Flask-powered backend for prediction, evaluation, training, and chatbot responses.

---

## 🗂 Project Structure

```

thyroid\_cancer\_project/
├── app.py                   # Flask backend
├── templates/               # HTML templates for Flask (chatbot, prediction)
├── frontend/                # Node.js app (login/register UI)
│   └── server.js            # Node server
├── classifier/              # Training scripts for classification
├── models/                  # CNN and UNet model definitions
├── utils/                   # Utility functions
├── uploads/                # Uploaded images (runtime)
├── data/                    # Training and validation data (excluded from GitHub)
├── intents.json             # Chatbot intents and responses
├── cnn\_classifier.pth       # Trained CNN model
├── \*.pth                    # Other model checkpoints
├── app.log                  # Application logs
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

````

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/shilpan16/Thyroid-Cancer-Detector.git
cd Thyroid-Cancer-Detector
````

---

### Step 2: Run Flask Backend

```bash
# (Create a virtual environment if needed)
pip install -r requirements.txt
python app.py
```

This runs the backend on:
👉 `http://127.0.0.1:5000`

---

### Step 3: Run Node.js Frontend (Login/Register)

```bash
cd frontend
npm install
node server.js
```

This runs the frontend on:
👉 `http://localhost:3018/login`

> The frontend communicates with the Flask backend via REST APIs (e.g., for login, prediction, chatbot, etc.)

---

## 🧠 Model Training

To train the classifier:

```bash
python train_classifier.py
```

For U-Net segmentation:

```bash
python train_seg.py
```

---

## 💬 Chatbot API (via Flask)

Send a POST request to:

```
POST /chatbot
{
  "message": "What are the symptoms of thyroid cancer?"
}
```

Returns an intelligent chatbot reply based on your intents.

---

## 📊 Evaluation

Use this endpoint to evaluate your model on the validation set:

```bash
GET /evaluate
```

Returns:

* Precision
* Recall
* F1-score

---

## 📎 Requirements

* Python 3.8+
* Flask
* torch, torchvision
* Pillow
* sentence-transformers
* scikit-learn
* Node.js (for frontend)

---
## 📷 Sample Results
### Login and Register pages
"C:\Users\Shilpa\Pictures\Screenshots\Screenshot 2025-05-19 224649.png"
"C:\Users\Shilpa\Pictures\Screenshots\Screenshot 2025-05-19 224843.png"

### 🔍 CNN Prediction (Benign vs Malignant)


### 💬 Chatbot Response
![Chatbot](results/chatbot_response.png)


## 👩‍💻 Author

**Shilpa Neralla**
Final Year B.Tech | AI & ML
Sphoorthy Engineering College

---

## 📜 License

This project is for educational and academic use only.

