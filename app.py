import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nltk()

#  CSS Styling 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-dark:    #0a0a0f;
    --bg-card:    #12121a;
    --bg-glass:   rgba(255,255,255,0.04);
    --accent:     #ff6b6b;
    --accent2:    #ffd93d;
    --accent3:    #6bcb77;
    --accent4:    #4d96ff;
    --text:       #f0f0f0;
    --muted:      #888;
    --border:     rgba(255,255,255,0.08);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #1a0a2e 0%, #0a0a0f 40%),
                radial-gradient(ellipse at 80% 80%, #0d1a2e 0%, transparent 50%);
    background-color: #0a0a0f;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Syne', sans-serif !important; font-size: 2rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #ff8e53);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(255,107,107,0.3);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255,107,107,0.5);
}

/* ── Text Area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(255,107,107,0.2) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-glass);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 20px;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: var(--muted);
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 8px 8px 0 0;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-glass) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 10px; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Radio ── */
.stRadio label { color: var(--text) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--muted) !important; font-size: 0.85rem; }

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-glass) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Custom Cards ── */
.result-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.result-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.15); }

.positive-card { border-left: 4px solid var(--accent3) !important; }
.negative-card  { border-left: 4px solid var(--accent) !important; }
.neutral-card   { border-left: 4px solid var(--muted) !important; }

.score-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.5px;
}
.badge-pos { background: rgba(107,203,119,0.15); color: #6bcb77; border: 1px solid rgba(107,203,119,0.3); }
.badge-neg { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); }
.badge-neu { background: rgba(136,136,136,0.15); color: #aaa;    border: 1px solid rgba(136,136,136,0.3); }

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.95rem;
}
.stat-label { color: var(--muted); }
.stat-value { color: var(--text); font-weight: 500; font-family: 'Syne', sans-serif; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.1rem;
    margin-top: 8px;
    font-weight: 300;
}
.tag {
    display: inline-block;
    background: rgba(255,107,107,0.1);
    color: var(--accent);
    border: 1px solid rgba(255,107,107,0.2);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


#  Helper Functions 
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s\.!\?,\'\":-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_textblob(text):
    score = TextBlob(text).sentiment.polarity
    subj  = TextBlob(text).sentiment.subjectivity
    label = 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
    return score, subj, label

def get_vader(text):
    scores = sia.polarity_scores(text)
    score  = scores['compound']
    label  = 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
    return score, scores, label

def sentiment_emoji(label):
    return {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(label, '❓')

def sentiment_color(label):
    return {'positive': '#6bcb77', 'negative': '#ff6b6b', 'neutral': '#aaa'}.get(label, '#fff')

def make_gauge(score, title, color):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.patch.set_facecolor('#12121a')
    ax.set_facecolor('#12121a')
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#222', linewidth=18, solid_capstyle='round')
    filled = np.linspace(np.pi, np.pi - score * np.pi, 200)
    ax.plot(np.cos(filled), np.sin(filled), color=color, linewidth=18, solid_capstyle='round')
    ax.text(0, -0.15, f'{score:.2f}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='white', fontfamily='monospace')
    ax.text(0, -0.5, title, ha='center', va='center', fontsize=10, color='#888')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.8, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


#  Sidebar 
with st.sidebar:
    st.markdown('<div class="tag">🎬 Sentiment AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:#f0f0f0;">Movie Review<br>Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#888;font-size:0.85rem;">TextBlob & VADER algorithms</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(255,255,255,0.08)">', unsafe_allow_html=True)

    mode = st.radio("**Analysis Mode**", ["✍️ Single Text", "📊 CSV Dataset"])

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08)">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#666;line-height:1.8">
    <b style="color:#aaa">TextBlob</b><br>Rule-based polarity scoring<br><br>
    <b style="color:#aaa">VADER</b><br>Social media optimized NLP<br><br>
    <b style="color:#aaa">Score Range</b><br>-1.0 (negative) → +1.0 (positive)
    </div>
    """, unsafe_allow_html=True)


#  Main Content 
st.markdown("""
<div style="padding: 40px 0 20px 0">
    <p class="hero-title">Sentiment<br>Analysis</p>
    <p class="hero-sub">Analyze movie reviews using TextBlob & VADER — instantly</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)



if mode == "✍️ Single Text":

    st.markdown("### ✍️ Enter Your Review")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_text = st.text_area(
            "Type or paste any movie review:",
            placeholder="e.g. This movie was absolutely brilliant! The acting was superb and the storyline kept me engaged throughout...",
            height=160,
            label_visibility="collapsed"
        )

        analyze_btn = st.button("🔍 Analyze Sentiment")

    with col2:
        st.markdown("""
        <div class="result-card" style="height:185px">
            <p style="color:#888;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin:0 0 12px 0">Quick Tips</p>
            <p style="color:#ccc;font-size:0.9rem;line-height:1.8;margin:0">
            💡 Works best with <b style="color:#fff">full sentences</b><br>
            💡 Supports <b style="color:#fff">long reviews</b><br>
            💡 Try <b style="color:#fff">mixed sentiments</b><br>
            💡 HTML tags are <b style="color:#fff">auto-cleaned</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn and user_text.strip():
        cleaned   = clean_text(user_text)
        tb_score, tb_subj, tb_label   = get_textblob(cleaned)
        vd_score, vd_scores, vd_label = get_vader(cleaned)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Results")

        
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("TextBlob Score",  f"{tb_score:+.3f}")
        with m2: st.metric("VADER Score",     f"{vd_score:+.3f}")
        with m3: st.metric("Subjectivity",    f"{tb_subj:.3f}")
        with m4: st.metric("Word Count",      len(user_text.split()))

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        # TextBlob card
        with r1:
            badge_class = f"badge-{'pos' if tb_label=='positive' else 'neg' if tb_label=='negative' else 'neu'}"
            card_class  = f"{'positive' if tb_label=='positive' else 'negative' if tb_label=='negative' else 'neutral'}-card"
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                    <span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700">TextBlob</span>
                    <span class="score-badge {badge_class}">{sentiment_emoji(tb_label)} {tb_label.upper()}</span>
                </div>
                <div class="stat-row"><span class="stat-label">Polarity Score</span><span class="stat-value">{tb_score:+.4f}</span></div>
                <div class="stat-row"><span class="stat-label">Subjectivity</span><span class="stat-value">{tb_subj:.4f}</span></div>
                <div class="stat-row" style="border:none"><span class="stat-label">Method</span><span class="stat-value">Rule-based NLP</span></div>
            </div>
            """, unsafe_allow_html=True)

        # VADER card
        with r2:
            badge_class = f"badge-{'pos' if vd_label=='positive' else 'neg' if vd_label=='negative' else 'neu'}"
            card_class  = f"{'positive' if vd_label=='positive' else 'negative' if vd_label=='negative' else 'neutral'}-card"
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                    <span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700">VADER</span>
                    <span class="score-badge {badge_class}">{sentiment_emoji(vd_label)} {vd_label.upper()}</span>
                </div>
                <div class="stat-row"><span class="stat-label">Compound Score</span><span class="stat-value">{vd_score:+.4f}</span></div>
                <div class="stat-row"><span class="stat-label">Positive</span><span class="stat-value">{vd_scores['pos']:.4f}</span></div>
                <div class="stat-row"><span class="stat-label">Negative</span><span class="stat-value">{vd_scores['neg']:.4f}</span></div>
                <div class="stat-row" style="border:none"><span class="stat-label">Neutral</span><span class="stat-value">{vd_scores['neu']:.4f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Agreement check
        st.markdown("<br>", unsafe_allow_html=True)
        if tb_label == vd_label:
            st.success(f"✅ Both algorithms **agree**: This review is **{tb_label.upper()}** {sentiment_emoji(tb_label)}")
        else:
            st.warning(f"⚠️ Algorithms **disagree** — TextBlob says **{tb_label}**, VADER says **{vd_label}**")

        # Gauge charts
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Score Visualization")
        g1, g2, g3 = st.columns(3)

        tb_norm = (tb_score + 1) / 2
        vd_norm = (vd_score + 1) / 2

        with g1:
            fig = make_gauge(tb_norm, "TextBlob Polarity", sentiment_color(tb_label))
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with g2:
            fig = make_gauge(vd_norm, "VADER Compound", sentiment_color(vd_label))
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with g3:
            fig = make_gauge(tb_subj, "Subjectivity", "#4d96ff")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze!")



else:
    st.markdown("### 📂 Upload Your Dataset")

    uploaded = st.file_uploader(
        "Upload IMDB Dataset CSV",
        type=['csv'],
        help="Needs 'review' and 'sentiment' columns"
    )

    if uploaded:
        with st.spinner("⏳ Loading dataset..."):
            df = pd.read_csv(uploaded)

        st.success(f"✅ Loaded **{len(df):,}** reviews!")

        # Preview
        with st.expander("👀 Preview Dataset", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        # Sample size slider
        st.markdown("#### ⚙️ Analysis Settings")
        sample_size = st.slider(
            "Sample size (larger = slower but more accurate)",
            min_value=100,
            max_value=min(5000, len(df)),
            value=min(1000, len(df)),
            step=100
        )

        run_btn = st.button("🚀 Run Full Analysis")

        if run_btn:
            df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            # ── Progress ──
            progress = st.progress(0)
            status   = st.empty()

            status.text("🧹 Cleaning text...")
            df_sample['clean_review'] = df_sample['review'].apply(clean_text)
            progress.progress(20)

            status.text("🔵 Running TextBlob...")
            results = df_sample['clean_review'].apply(get_textblob)
            df_sample['tb_score']     = results.apply(lambda x: x[0])
            df_sample['tb_subj']      = results.apply(lambda x: x[1])
            df_sample['tb_sentiment'] = results.apply(lambda x: x[2])
            progress.progress(60)

            status.text("🟣 Running VADER...")
            results = df_sample['clean_review'].apply(get_vader)
            df_sample['vader_score']     = results.apply(lambda x: x[0])
            df_sample['vader_sentiment'] = results.apply(lambda x: x[2])
            progress.progress(90)

            status.text("📊 Generating charts...")

            #  Accuracy 
            def accuracy(pred_col):
                mask    = df_sample[pred_col] != 'neutral'
                subset  = df_sample[mask]
                return (subset[pred_col] == subset['sentiment']).mean() * 100

            tb_acc    = accuracy('tb_sentiment')
            vader_acc = accuracy('vader_sentiment')
            progress.progress(100)
            status.empty()

            #  Metrics 
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### 📊 Results Overview")

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Reviews Analyzed", f"{sample_size:,}")
            with m2: st.metric("TextBlob Accuracy", f"{tb_acc:.1f}%")
            with m3: st.metric("VADER Accuracy",    f"{vader_acc:.1f}%")
            with m4:
                winner = "TextBlob" if tb_acc > vader_acc else "VADER"
                st.metric("Best Performer", winner)

            #  Charts 
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Scores", "🎯 Accuracy", "🔥 Confusion"])

            plt.style.use('dark_background')
            DARK = '#0a0a0f'
            CARD = '#12121a'

            # Tab 1
            with tab1:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.patch.set_facecolor(DARK)
                cols_data = [('Original', 'sentiment'), ('TextBlob', 'tb_sentiment'), ('VADER', 'vader_sentiment')]
                cmap = {'positive': '#6bcb77', 'negative': '#ff6b6b', 'neutral': '#888'}

                for ax, (title, col) in zip(axes, cols_data):
                    ax.set_facecolor(CARD)
                    counts = df_sample[col].value_counts()
                    colors = [cmap.get(s, '#4d96ff') for s in counts.index]
                    bars   = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='none')
                    for bar, val in zip(bars, counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                                f'{val:,}', ha='center', fontsize=10, color='white', fontweight='bold')
                    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)
                    ax.set_facecolor(CARD); ax.tick_params(colors='#888')
                    for spine in ax.spines.values(): spine.set_visible(False)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Tab 2
            with tab2:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.patch.set_facecolor(DARK)
                for ax, (col, title) in zip(axes, [('tb_score','TextBlob Polarity'), ('vader_score','VADER Compound')]):
                    ax.set_facecolor(CARD)
                    ax.hist(df_sample[df_sample['sentiment']=='positive'][col], bins=40, alpha=0.7, color='#6bcb77', label='Positive')
                    ax.hist(df_sample[df_sample['sentiment']=='negative'][col], bins=40, alpha=0.7, color='#ff6b6b', label='Negative')
                    ax.axvline(0, color='white', linestyle='--', lw=1, alpha=0.5)
                    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
                    ax.tick_params(colors='#888')
                    ax.legend(facecolor=DARK, labelcolor='white')
                    for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Tab 3
            with tab3:
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
                bars = ax.bar(['TextBlob', 'VADER'], [tb_acc, vader_acc],
                              color=['#4d96ff', '#ffd93d'], width=0.4, edgecolor='none')
                for bar, acc in zip(bars, [tb_acc, vader_acc]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 4,
                            f'{acc:.1f}%', ha='center', va='top',
                            fontsize=18, fontweight='bold', color='#0a0a0f')
                ax.axhline(50, color='#555', linestyle='--', lw=1, label='Baseline (50%)')
                ax.set_title('Accuracy Comparison', fontsize=13, fontweight='bold', color='white')
                ax.set_ylim(0, 105); ax.tick_params(colors='#888')
                ax.legend(facecolor=DARK, labelcolor='white')
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Tab 4
            with tab4:
                from sklearn.metrics import confusion_matrix
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor(DARK)
                for ax, col, title in zip(axes, ['tb_sentiment','vader_sentiment'], ['TextBlob','VADER']):
                    mask   = df_sample[col] != 'neutral'
                    subset = df_sample[mask]
                    cm     = confusion_matrix(subset['sentiment'], subset[col], labels=['negative','positive'])
                    sns.heatmap(cm, annot=True, fmt=',', cmap='RdYlGn', ax=ax,
                                xticklabels=['Pred: Neg','Pred: Pos'],
                                yticklabels=['Actual: Neg','Actual: Pos'],
                                linewidths=2, linecolor=DARK)
                    ax.set_title(f'{title} Confusion Matrix', fontsize=12, fontweight='bold', color='white')
                    ax.tick_params(colors='#888')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### 💾 Download Results")
            csv = df_sample[['review','sentiment','tb_score','tb_sentiment','vader_score','vader_sentiment']].to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

    else:
        st.markdown("""
        <div class="result-card" style="text-align:center;padding:48px">
            <p style="font-size:3rem;margin:0">📁</p>
            <p style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;margin:12px 0 6px">Upload your CSV file</p>
            <p style="color:#666;margin:0">Needs columns: <code style="color:#ff6b6b">review</code> and <code style="color:#ff6b6b">sentiment</code></p>
        </div>
        """, unsafe_allow_html=True)
