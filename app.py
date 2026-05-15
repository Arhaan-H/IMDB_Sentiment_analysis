# IMDB_Sentiment_Analysis
import streamlit as st
import pickle, re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0d1117;color:#e6edf3;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#161b22;border-radius:12px;
  padding:6px;border:1px solid #30363d;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8b949e;font-weight:500;
  padding:8px 20px;background:transparent;border:none;}
.stTabs [aria-selected="true"]{background:#1f6feb!important;color:#fff!important;}
.metric-card{background:linear-gradient(135deg,#161b22,#1c2128);border:1px solid #30363d;
  border-radius:12px;padding:20px;text-align:center;transition:transform .2s,border-color .2s;}
.metric-card:hover{transform:translateY(-3px);border-color:#1f6feb;}
.metric-value{font-size:2rem;font-weight:700;color:#1f6feb;}
.metric-label{font-size:.85rem;color:#8b949e;margin-top:4px;}
.section-header{font-size:1.3rem;font-weight:600;color:#e6edf3;
  border-left:4px solid #1f6feb;padding-left:12px;margin:24px 0 16px;}
.review-card{background:#161b22;border:1px solid #30363d;border-radius:10px;
  padding:16px;margin-bottom:12px;}
.badge-pos{background:#0d4f1f;color:#3fb950;padding:3px 12px;
  border-radius:20px;font-size:.8rem;font-weight:600;}
.badge-neg{background:#4f0d1c;color:#f85149;padding:3px 12px;
  border-radius:20px;font-size:.8rem;font-weight:600;}
.info-banner{background:linear-gradient(90deg,#1f3a5f,#162a43);
  border:1px solid #1f6feb;border-radius:10px;padding:14px 20px;
  color:#79c0ff;font-size:.9rem;margin-bottom:24px;}
.best-badge{background:linear-gradient(90deg,#0d4f1f,#0a3d17);border:2px solid #3fb950;
  border-radius:12px;padding:16px 24px;display:inline-block;margin-bottom:20px;}
.pred-positive{background:linear-gradient(135deg,#0d4f1f,#0a3d17);
  border:2px solid #3fb950;border-radius:16px;padding:28px;text-align:center;}
.pred-negative{background:linear-gradient(135deg,#4f0d1c,#3d0a14);
  border:2px solid #f85149;border-radius:16px;padding:28px;text-align:center;}
.pred-label{font-size:2.5rem;font-weight:700;margin-bottom:8px;}
.pred-conf{font-size:1.1rem;color:#8b949e;}
.word-chip{display:inline-block;background:#1f3a5f;border:1px solid #1f6feb;
  color:#79c0ff;border-radius:20px;padding:4px 14px;margin:4px;font-size:.85rem;}
button[kind="primary"]{background:linear-gradient(90deg,#1f6feb,#388bfd)!important;
  border:none!important;border-radius:8px!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.title.set_color("#e6edf3")
    return fig, ax

STOP_WORDS = set(stopwords.words("english"))
negations = {"not","no","nor","ain","aren","couldn","didn","doesn","hadn",
             "hasn","haven","isn","mightn","mustn","needn","shan","shouldn",
             "wasn","weren","won","wouldn"}
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

# ── Load pickles ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_pickles():
    missing = [f for f in ["vectorizer.pkl","best_model.pkl","model_results.pkl"]
               if not os.path.exists(f)]
    if missing:
        return None, None, None
    with open("vectorizer.pkl",   "rb") as f: vec = pickle.load(f)
    with open("best_model.pkl",   "rb") as f: mdl = pickle.load(f)
    with open("model_results.pkl","rb") as f: res = pickle.load(f)
    return vec, mdl, res

vectorizer, model, results = load_pickles()

if vectorizer is None:
    st.error("⚠️ Pickle files not found. Please run:  `python train.py`  first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:30px 0 10px'>
  <h1 style='font-size:2.6rem;font-weight:700;
      background:linear-gradient(90deg,#1f6feb,#58a6ff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      margin-bottom:6px'>🎬 IMDB Sentiment Analysis</h1>
  <p style='color:#8b949e;font-size:1rem'>
    Full ML pipeline · Logistic Regression · Naive Bayes · Linear SVM
  </p>
</div>""", unsafe_allow_html=True)

best_name = results["best_model_name"]
best_f1   = results["metrics"][best_name]["F1-Score"]

st.markdown(f"""
<div class='info-banner'>
  ℹ️ <b>Loaded from IMDB_Dataset.csv</b> — {results['total_reviews']:,} reviews
  ({results['class_counts']['positive']:,} positive /
   {results['class_counts']['negative']:,} negative) ·
  TF-IDF (10K features, unigrams+bigrams) · 80/20 stratified split ·
  🏆 Best: <b>{best_name}</b> (F1 = {best_f1:.4f})
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab4, tab1, tab2, tab3 = st.tabs([
    "🔮 Live Predictor", "📊 Data Overview",
    "🧹 Preprocessing",  "🤖 Model Results"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    pos = results["class_counts"]["positive"]
    neg = results["class_counts"]["negative"]
    total = results["total_reviews"]

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{total:,}",  "Total Reviews"),
        (c2, "2",           "Columns"),
        (c3, f"{pos:,}",    "Positive ✅"),
        (c4, f"{neg:,}",    "Negative ❌"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Class Distribution & Review Lengths</div>",
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = dark_fig(5, 3.8)
        bars = ax.bar(["Positive","Negative"], [pos, neg],
                      color=["#3fb950","#f85149"], width=0.5, edgecolor="#30363d")
        for bar, v in zip(bars, [pos, neg]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150,
                    f"{v:,}", ha="center", color="#e6edf3", fontsize=10)
        ax.set_title("Class Balance", fontsize=12)
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(pos,neg)*1.15)
        ax.grid(axis="y", color="#30363d", linewidth=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = dark_fig(5, 3.8)
        ax.hist(results["rl_pos"], bins=60, color="#3fb950", alpha=0.65,
                label="Positive", edgecolor="#30363d")
        ax.hist(results["rl_neg"], bins=60, color="#f85149", alpha=0.65,
                label="Negative", edgecolor="#30363d")
        ax.set_title("Review Length Distribution (words)", fontsize=12)
        ax.set_xlabel("Words per review"); ax.set_ylabel("Count")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        ax.grid(axis="y", color="#30363d", linewidth=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Sample Reviews</div>",
                unsafe_allow_html=True)
    for _, row in results["sample_reviews"].iterrows():
        badge = "badge-pos" if row.sentiment == "positive" else "badge-neg"
        label = "POSITIVE ✅" if row.sentiment == "positive" else "NEGATIVE ❌"
        snippet = str(row["review"])[:320].replace("<br />", " ") + "…"
        st.markdown(f"""
        <div class='review-card'>
          <span class='{badge}'>{label}</span>
          <p style='margin-top:10px;color:#c9d1d9;font-size:.9rem;line-height:1.6'>
            {snippet}</p>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Before vs After Cleaning</div>",
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**🔴 Raw Review**")
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #f8514944;border-radius:10px;
                    padding:16px;font-size:.85rem;color:#c9d1d9;line-height:1.7;
                    min-height:180px;overflow:auto'>
            {str(results["sample_raw"])[:500]}…
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("**🟢 Cleaned Review**")
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #3fb95044;border-radius:10px;
                    padding:16px;font-size:.85rem;color:#c9d1d9;line-height:1.7;
                    min-height:180px;overflow:auto'>
            {str(results["sample_clean"])[:500]}…
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Preprocessing Steps Applied</div>",
                unsafe_allow_html=True)
    steps = [
        ("1️⃣","Lowercase","All text → lowercase"),
        ("2️⃣","Remove HTML","Strip `<br />`, `<b>` etc."),
        ("3️⃣","Remove Punctuation","Keep only a-z letters"),
        ("4️⃣","Remove Stopwords","Drop common words (keep negations)"),
        ("5️⃣","Lemmatization","Reduce to base form"),
    ]
    cols = st.columns(5)
    for col, (num, name, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;
                    padding:14px;text-align:center;height:110px'>
          <div style='font-size:1.4rem'>{num}</div>
          <div style='font-weight:600;color:#58a6ff;font-size:.88rem;margin:5px 0'>{name}</div>
          <div style='color:#8b949e;font-size:.76rem'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Top 20 Words — Positive vs Negative</div>",
                unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        words, counts = zip(*results["top_words_pos"])
        fig, ax = dark_fig(5, 5.5)
        ax.barh(list(words)[::-1], list(counts)[::-1], color="#3fb950")
        ax.set_title("Top 20 — Positive Reviews", fontsize=11)
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", color="#30363d", linewidth=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with p2:
        words, counts = zip(*results["top_words_neg"])
        fig, ax = dark_fig(5, 5.5)
        ax.barh(list(words)[::-1], list(counts)[::-1], color="#f85149")
        ax.set_title("Top 20 — Negative Reviews", fontsize=11)
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", color="#30363d", linewidth=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class='best-badge'>
      🏆 <b style='color:#3fb950'>Best Model: {best_name}</b> &nbsp;·&nbsp;
      F1-Score: <b style='color:#3fb950'>{best_f1:.4f}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Model Comparison</div>",
                unsafe_allow_html=True)
    rows_html = ""
    for name, m in results["metrics"].items():
        cls = "best-row" if name == best_name else ""
        star = " ⭐" if name == best_name else ""
        rows_html += f"""<tr class='{cls}'>
          <td>{name}{star}</td><td>{m['Accuracy']:.4f}</td>
          <td>{m['Precision']:.4f}</td><td>{m['Recall']:.4f}</td>
          <td><b>{m['F1-Score']:.4f}</b></td></tr>"""

    st.markdown(f"""
    <style>
      .styled-table{{width:100%;border-collapse:collapse;background:#161b22;
        border-radius:10px;overflow:hidden;}}
      .styled-table th{{background:#1f6feb22;color:#79c0ff;padding:12px 16px;
        text-align:left;font-weight:600;}}
      .styled-table td{{padding:12px 16px;border-top:1px solid #30363d;color:#e6edf3;}}
      .styled-table tr:hover td{{background:#1c2128;}}
      .best-row td{{background:#0d4f1f22!important;}}
    </style>
    <table class='styled-table'><thead><tr>
      <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th>
    </tr></thead><tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown(f"<div class='section-header'>Confusion Matrix — {best_name}</div>",
                    unsafe_allow_html=True)
        cm = results["confusion_matrix"]
        fig, ax = dark_fig(4.5, 3.8)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative","Positive"],
                    yticklabels=["Negative","Positive"], ax=ax,
                    linewidths=0.5, linecolor="#30363d",
                    annot_kws={"size":14,"color":"white"})
        ax.set_xlabel("Predicted",color="#8b949e")
        ax.set_ylabel("Actual",color="#8b949e")
        ax.tick_params(colors="#8b949e")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_roc:
        st.markdown("<div class='section-header'>ROC Curves (All Models)</div>",
                    unsafe_allow_html=True)
        colors = {"Logistic Regression":"#1f6feb",
                  "Multinomial Naive Bayes":"#3fb950",
                  "Linear SVM":"#f85149"}
        fig = go.Figure()
        fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                      line=dict(color="#30363d",dash="dash"))
        for name, rd in results["roc_data"].items():
            fig.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"],
                name=f"{name} (AUC={rd['auc']:.3f})",
                line=dict(color=colors.get(name,"#58a6ff"),width=2.5)))
        fig.update_layout(
            plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
            xaxis=dict(title="False Positive Rate",color="#8b949e",gridcolor="#30363d"),
            yaxis=dict(title="True Positive Rate",color="#8b949e",gridcolor="#30363d"),
            legend=dict(bgcolor="#161b22",font=dict(color="#e6edf3")),
            font=dict(color="#e6edf3"), height=360,
            margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:12px;
                padding:16px 20px;margin-bottom:20px'>
      <p style='color:#8b949e;margin:0'>Using <b style='color:#58a6ff'>{best_name}</b>
      &nbsp;·&nbsp; F1-Score: <b style='color:#3fb950'>{best_f1:.4f}</b></p>
    </div>""", unsafe_allow_html=True)

    user_review = st.text_area("✍️ Enter a movie review",
                               height=160,
                               placeholder="Type or paste any movie review here…",
                               key="review_input")

    if st.button("🔮 Predict Sentiment", type="primary"):
        if user_review.strip():
            cleaned = clean_text(user_review)
            vec_input = vectorizer.transform([cleaned])

            # Confidence score
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vec_input)[0]
                pred = int(np.argmax(prob))
                conf = float(prob[pred]) * 100
            else:
                raw_score = model.decision_function(vec_input)[0]
                pred = int(raw_score > 0)
                conf = min(99.9, 50 + abs(raw_score) * 5)

            # Result badge
            if pred == 1:
                st.markdown(f"""
                <div class='pred-positive'>
                  <div class='pred-label' style='color:#3fb950'>😊 POSITIVE</div>
                  <div class='pred-conf'>Confidence: <b>{conf:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-negative'>
                  <div class='pred-label' style='color:#f85149'>😞 NEGATIVE</div>
                  <div class='pred-conf'>Confidence: <b>{conf:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)

            # Top 5 influential words
            st.markdown("<div class='section-header' style='margin-top:20px'>"
                        "Top 5 Influential Words</div>", unsafe_allow_html=True)
            try:
                feature_names = results.get("feature_names", [])
                if feature_names and hasattr(model, "coef_"):
                    coefs = np.array(model.coef_[0])
                elif feature_names and hasattr(model, "feature_log_prob_"):
                    coefs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                else:
                    coefs = None

                if coefs is not None:
                    vec_arr = vec_input.toarray()[0]
                    # Only consider words present in the review
                    present = np.where(vec_arr > 0)[0]
                    if len(present) > 0:
                        scores = vec_arr[present] * coefs[present]
                        top_idx = present[np.argsort(np.abs(scores))[::-1][:5]]
                        chips_html = ""
                        for idx in top_idx:
                            word = feature_names[idx]
                            sentiment_color = "#3fb950" if coefs[idx] > 0 else "#f85149"
                            chips_html += (f"<span class='word-chip' "
                                           f"style='border-color:{sentiment_color};"
                                           f"color:{sentiment_color}'>"
                                           f"{word}</span>")
                        st.markdown(f"<div style='margin-top:8px'>{chips_html}</div>",
                                    unsafe_allow_html=True)
                        st.caption("🟢 Green = pushes toward positive · 🔴 Red = pushes toward negative")
            except Exception:
                pass  # silently skip if anything goes wrong

            with st.expander("🔍 Cleaned text sent to model"):
                st.code(cleaned or "(empty after cleaning)", language="text")
        else:
            st.warning("Please enter a review first.")

    # Quick examples
    st.markdown("<div class='section-header'>Try These Examples</div>",
                unsafe_allow_html=True)
    ex_pos = ("An absolute masterpiece. The performances were breathtaking, "
              "the story deeply moving. One of the best films I have ever seen.")
    ex_neg = ("A complete waste of time. The plot made no sense, acting was wooden, "
              "and the ending was deeply disappointing. Cannot recommend this film.")
    c1, c2 = st.columns(2)
    if c1.button("📋 Positive example", use_container_width=True):
        st.session_state["review_input"] = ex_pos
        st.rerun()
    if c2.button("📋 Negative example", use_container_width=True):
        st.session_state["review_input"] = ex_neg
        st.rerun()
