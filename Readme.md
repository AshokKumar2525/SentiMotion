# ✨ EmotiVerse – BERT Twitter Sentiment/Emotion Analyzer

EmotiVerse is an interactive **Streamlit web app** that analyzes the emotional tone of tweets
using a **fine-tuned BERT (bert-base-uncased)** model.  
It predicts one of six emotions:

> 😢 **Sadness** • 😀 **Joy** • ❤️ **Love** • 😡 **Anger** • 😨 **Fear** • 😮 **Surprise**

---

## 🌟 Demo
![demo-screenshot](docs/demo.png)  
(Type a tweet → Click **Analyze** → See predicted emotion with confidence bars.)

---

## 🧩 Features
- **6-class emotion detection** powered by a fine-tuned BERT model
- Beautiful **gradient UI** with animated background
- Real-time probability bars with **percentages next to each label**
- GPU acceleration (if available)

---

## 📂 Repository Structure
```
├── app.py # Streamlit app
├── requirements.txt # Python dependencies
├── bert-base-uncased-sentiment-model/
│ ├── config.json
│ ├── model.safetensors
│ ├── vocab.txt
│ └── ... (tokenizer + training args)
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
### 2️⃣ Create & Activate Virtual Environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# .\venv\Scripts\activate         # Windows (PowerShell)
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the App
```bash
streamlit run app.py
```

---

## 💡 Model Details

- Base model: bert-base-uncased
- Classes (id → label):
```
0 → Sadness
1 → Joy
2 → Love
3 → Anger
4 → Fear
5 → Surprise
```
- Fine-tuned on a custom Twitter multi-class sentiment dataset.
