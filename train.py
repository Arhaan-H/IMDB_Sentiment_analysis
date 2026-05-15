# IMDB_Sentiment_Analysis
import os, re, pickle, warnings
import pandas as pd
import numpy as np
from collections import Counter

import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
warnings.filterwarnings("ignore")

STOP_WORDS = set(stopwords.words("english"))
negations = {"not", "no", "nor", "ain", "aren", "couldn", "didn", "doesn",
             "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn",
             "shan", "shouldn", "wasn", "weren", "won", "wouldn"}
STOP_WORDS -= negations
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

for fname in ["IMDB_Dataset.csv", "IMDB Dataset.csv"]:
    if os.path.exists(fname):
        CSV_PATH = fname
        break
else:
    raise FileNotFoundError(
        "CSV not found. Place 'IMDB_Dataset.csv' in the project folder.")

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, engine="python")
df.columns = [c.lower().strip() for c in df.columns]
df["sentiment"] = df["sentiment"].str.lower().str.strip()
df = df[df["sentiment"].isin(["positive", "negative"])].dropna().reset_index(drop=True)
df["review"] = df["review"].astype(str)
df["label"]  = (df["sentiment"] == "positive").astype(int)
print(f"  {len(df):,} reviews - {df.label.sum():,} positive, {(~df.label.astype(bool)).sum():,} negative")

print("Cleaning text...")
df["clean"] = df["review"].apply(clean_text)

print("Vectorizing...")
X = list(df["clean"])
y = np.array(df["label"].tolist(), dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
X_tr = tfidf.fit_transform(X_train)
X_te = tfidf.transform(X_test)

print("Training models...")
model_defs = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Linear SVM": SGDClassifier(loss="modified_huber", random_state=42, max_iter=1000),
}

metrics, proba_scores = {}, {}
trained_models = {}

for name, model in model_defs.items():
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    trained_models[name] = model
    metrics[name] = {
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test, preds), 4),
        "F1-Score":  round(f1_score(y_test, preds), 4),
    }
    if hasattr(model, "predict_proba"):
        proba_scores[name] = model.predict_proba(X_te)[:, 1]
    else:
        s = model.decision_function(X_te)
        proba_scores[name] = (s - s.min()) / (s.max() - s.min() + 1e-9)

best_name = max(metrics, key=lambda k: metrics[k]["F1-Score"])
best_model = trained_models[best_name]
best_preds  = best_model.predict(X_te)

print("\n-- Model Comparison " + "-"*37)
print(f"{'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 56)
for name, m in metrics.items():
    star = " *" if name == best_name else ""
    print(f"{name+star:<30} {m['Accuracy']:>6.4f} {m['Precision']:>6.4f} "
          f"{m['Recall']:>6.4f} {m['F1-Score']:>6.4f}")
print(f"\nBest model: {best_name} (F1 = {metrics[best_name]['F1-Score']:.4f})")

roc_data = {}
for name, scores in proba_scores.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                      "auc": round(auc(fpr, tpr), 4)}

cm = confusion_matrix(y_test, best_preds)

def top_words(df_sub, n=20):
    corpus = " ".join(df_sub["clean"].values)
    return Counter(corpus.split()).most_common(n)

top_pos = top_words(df[df.sentiment == "positive"])
top_neg = top_words(df[df.sentiment == "negative"])

sample_df = pd.concat([
    df[df.sentiment == "positive"].sample(3, random_state=1),
    df[df.sentiment == "negative"].sample(2, random_state=1),
]).sample(frac=1, random_state=7)[["review", "sentiment"]].reset_index(drop=True)

df["rl"] = df["review"].apply(lambda t: len(t.split()))
rl_pos = df[df.sentiment == "positive"]["rl"].clip(0, 1000).tolist()
rl_neg = df[df.sentiment == "negative"]["rl"].clip(0, 1000).tolist()

sample_idx = 42
sample_raw   = df["review"].iloc[sample_idx]
sample_clean = df["clean"].iloc[sample_idx]

class_counts = {"positive": int((df.sentiment == "positive").sum()),
                "negative": int((df.sentiment == "negative").sum())}

feature_names = tfidf.get_feature_names_out().tolist()

model_results = {
    "best_model_name": best_name,
    "metrics":         metrics,
    "confusion_matrix": cm,
    "roc_data":        roc_data,
    "top_words_pos":   top_pos,
    "top_words_neg":   top_neg,
    "sample_reviews":  sample_df,
    "rl_pos":          rl_pos,
    "rl_neg":          rl_neg,
    "sample_raw":      sample_raw,
    "sample_clean":    sample_clean,
    "class_counts":    class_counts,
    "total_reviews":   len(df),
    "feature_names":   feature_names,
}

print("\nSaving pickle files...")
with open("vectorizer.pkl",   "wb") as f: pickle.dump(tfidf,        f)
with open("best_model.pkl",   "wb") as f: pickle.dump(best_model,   f)
with open("model_results.pkl","wb") as f: pickle.dump(model_results, f)

print("  [OK] vectorizer.pkl")
print("  [OK] best_model.pkl")
print("  [OK] model_results.pkl")
print("\nDone! Now run: streamlit run app.py")
