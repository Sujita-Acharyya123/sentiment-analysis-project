# 🎬 Movie Review Sentiment Analyzer

A beautiful Streamlit web app that analyzes movie review sentiments using **TextBlob** and **VADER** algorithms.

---

## 📸 Features

- ✍️ **Single Text Analysis** — Paste any review and get instant results
- 📊 **CSV Dataset Analysis** — Upload IMDB dataset and analyze thousands of reviews
- 🎯 **Dual Algorithm Comparison** — TextBlob vs VADER side by side
- 📈 **Visual Charts** — Distribution, Score Histogram, Accuracy, Confusion Matrix
- 💾 **Download Results** — Export analyzed data as CSV
- 🌙 **Dark Theme UI** — Beautiful modern design

---

## 🛠️ Installation

### Step 1 — Clone or Download the Project
```bash
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project
```

Or just download and put `app.py` and `requirements.txt` in a folder.

### Step 2 — Install Required Libraries
```bash
pip install -r requirements.txt
```

### Step 3 — Run the App
```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 📦 Requirements

```
streamlit
textblob
vaderSentiment
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
wordcloud
```

---

## 📁 Project Structure

```
sentiment_app/
│
├── app.py              ← Main Streamlit app
├── requirements.txt    ← All required libraries
└── README.md           ← This file
```

---

## 🚀 How to Use

### Single Text Mode
1. Select **✍️ Single Text** from sidebar
2. Type or paste any movie review
3. Click **🔍 Analyze Sentiment**
4. View results with TextBlob & VADER scores

### CSV Dataset Mode
1. Select **📊 CSV Dataset** from sidebar
2. Upload your `IMDB Dataset.csv`
3. Choose sample size using slider
4. Click **🚀 Run Full Analysis**
5. View charts and download results

---

## 📊 Algorithms Used

| Algorithm | Type | Best For |
|-----------|------|----------|
| **TextBlob** | Rule-based NLP | General text |
| **VADER** | Lexicon-based | Social media & reviews |

### Score Range
- **+1.0** = Very Positive 😊
- **0.0**  = Neutral 😐
- **-1.0** = Very Negative 😞

---

## 📋 Dataset Format

Due to large file size, the dataset is not included in this repository.

👉 You can use any dataset with the following columns:

review
sentiment

👉 Recommended dataset:
IMDB Movie Reviews Dataset (Kaggle)

---

## 👩‍💻 Built With

- [Streamlit](https://streamlit.io) — Web app framework
- [TextBlob](https://textblob.readthedocs.io) — NLP library
- [VADER](https://github.com/cjhutto/vaderSentiment) — Sentiment analysis
- [Matplotlib](https://matplotlib.org) — Charts
- [Seaborn](https://seaborn.pydata.org) — Heatmaps
- [Pandas](https://pandas.pydata.org) — Data processing

---

## 👩‍🎓 Project Info

- **Project Type:** Final Year Project
- **Topic:** Sentiment Analysis of IMDB Movie Reviews
- **Algorithms:** TextBlob + VADER

---

## 📝 License

This project is for educational purposes.
