# 🎬 Sentiment Analysis - Movie Reviews

A machine learning project that predicts whether a movie review is **Positive** or **Negative**.

## 📋 What This Project Does
- Loads 50,000 real IMDB movie reviews
- Cleans and processes the text data
- Trains 3 different ML models and compares them
- Includes a live predictor where you can type any review and get a prediction

## 📊 Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 88.76% ✅ (Best) |
| SVM | 87.97% |
| Naive Bayes | 85.06% |

## 🛠️ Libraries Used
- pandas, numpy - for data handling
- scikit-learn - for machine learning models
- nltk - for text cleaning
- matplotlib - for graphs

## 🚀 How To Run
1. Install libraries: `pip install pandas numpy scikit-learn nltk textblob matplotlib`
2. Run: `python main.py`
3. Type any movie review in the live predictor!
```

**Step 3 — Save the file** (Ctrl + S)

**Step 4 — Push it to GitHub:**
```
git add .
git commit -m "Added README file"
git push origin main