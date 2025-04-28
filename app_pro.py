
import streamlit as st
import pandas as pd
import numpy as np
import json, os
from sklearn.metrics import ndcg_score
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# — 1️⃣ Load & preprocess data
@st.cache_data
def load_data(path="ai_ml_datasets_210.csv"):
    df = pd.read_csv(path)
    # parse citations
    def parse_cit(v):
        v = str(v).replace("+","").upper()
        if "K" in v: return float(v.replace("K",""))*1_000
        if "M" in v: return float(v.replace("M",""))*1_000_000
        try: return float(v)
        except: return np.nan
    # parse size
    def parse_sz(v):
        try:
            p = str(v).split()[0].upper()
            if "B" in p: return float(p.replace("B",""))*1e9
            if "M" in p: return float(p.replace("M",""))*1e6
            if "K" in p: return float(p.replace("K",""))*1e3
            return float(p) if p.replace('.','',1).isdigit() else np.nan
        except:
            return np.nan

    df["Citations_Num"] = df["Citations"].apply(parse_cit)
    df["Size_Num"]      = df["Size"].apply(parse_sz)
    df["Diversity_Score"] = df["Diversity"].map({"Low":0.0,"Medium":0.5,"High":1.0})
    year_now = datetime.now().year
    df["Recency_Score"] = (df["Year"] - 2010) / (year_now - 2010)

    # metadata quality: fraction of non‐NaN on our four
    def meta_quality(r):
        score = 0
        for c in ["Citations_Num","Size_Num","Diversity_Score","Recency_Score"]:
            if not np.isnan(r[c]): score += 0.25
        return score
    df["Metadata_Quality_Score"] = df.apply(meta_quality, axis=1)

    # embed task types once
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df["Embedding"] = df["Task Type"].apply(lambda t: model.encode(t, convert_to_tensor=True))
    return df

df = load_data()

# — 2️⃣ UI: Title & query
st.title("📊 Personalized AI/ML Dataset Retrieval based on metadata")
raw_query = st.text_input("🔍 Describe your Task Type (e.g. 'image classification'):")
if not raw_query:
    st.info("Type something to start.")
    st.stop()

# — 3️⃣ Autocomplete suggestion (optional, reuses previous code)
all_tasks = sorted(df["Task Type"].unique())
suggestions = [t for t in all_tasks if raw_query.lower() in t.lower()]
if not suggestions:
    st.error("No close Task Type matches. Try rephrasing.")
    st.stop()
selected_task = st.selectbox("Or pick from known Task Types:", suggestions)

# — 4️⃣ Filter by selected_task
filtered = df[df["Task Type"] == selected_task]

# — 5️⃣ Sidebar: weight sliders for 5 criteria
st.sidebar.header("⚙️ Ranking Weights (sum needn’t be 1)")
w_size      = st.sidebar.slider("Size",              0.0, 1.0, 0.2)
w_cite      = st.sidebar.slider("Citations",         0.0, 1.0, 0.2)
w_div       = st.sidebar.slider("Diversity",         0.0, 1.0, 0.2)
w_recency   = st.sidebar.slider("Recency",           0.0, 1.0, 0.2)
w_sim       = st.sidebar.slider("Semantic Similarity",0.0,1.0,0.2)
weights = {
    "Size_Num":        w_size,
    "Citations_Num":   w_cite,
    "Diversity_Score": w_div,
    "Recency_Score":   w_recency,
    "Similarity":      w_sim
}
criteria = list(weights.keys())

# — 6️⃣ Compute semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
query_emb = model.encode(raw_query, convert_to_tensor=True)
filtered["Similarity"] = filtered["Embedding"].apply(lambda e: util.cos_sim(e, query_emb).item())

# — 7️⃣ TOPSIS ranking over the 5 criteria
def normalize(df, cols):
    tmp = df.copy()
    for c in cols:
        norm = np.sqrt((df[c]**2).sum())
        tmp[c] = df[c]/norm if norm>0 else 0
    return tmp

norm_df = normalize(filtered, criteria)
ideal_best  = {c: norm_df[c].max() for c in criteria}
ideal_worst = {c: norm_df[c].min() for c in criteria}

def topsis(row):
    sp = np.sqrt(sum(weights[c]*(row[c]-ideal_best[c])**2 for c in criteria))
    sn = np.sqrt(sum(weights[c]*(row[c]-ideal_worst[c])**2 for c in criteria))
    return sn/(sp+sn) if (sp+sn)>0 else 0

filtered["TOPSIS"]      = norm_df.apply(topsis, axis=1)
filtered["Final_Score"] = filtered["TOPSIS"] * filtered["Metadata_Quality_Score"]
ranked = filtered.sort_values("Final_Score", ascending=False)

# — 8️⃣ Show top10 + Source + feedback
st.subheader("🏆 Top 10 Datasets")
if "liked" not in st.session_state:
    st.session_state.liked = []

for idx, r in ranked.head(10).iterrows():
    st.markdown(f"**{r['Dataset Name']}**  —  _{r['Source']}_")
    st.write(f"- Task: {r['Task Type']} | Year: {r['Year']}")
    st.write(f"- Citations: {r['Citations']} | Size: {r['Size']}")
    st.write(f"- Diversity: {r['Diversity']} | Quality: {r['Metadata_Quality_Score']:.2f}")
    st.write(f"- Semantic Sim.: {r['Similarity']:.3f} | TOPSIS: {r['TOPSIS']:.3f}")
    st.write(f"- **Final Score:** {r['Final_Score']:.3f}")
    if st.button("👍 Like", key=idx):
        if r["Dataset Name"] not in st.session_state.liked:
            st.session_state.liked.append(r["Dataset Name"])
    st.markdown("---")

# — 9️⃣ Evaluation (NDCG@10, MRR)
if st.session_state.liked:
    top10 = ranked.head(10)
    preds = top10["Final_Score"].to_numpy().reshape(1,-1)
    rels  = [1 if nm in st.session_state.liked else 0 for nm in top10["Dataset Name"]]
    ndcg = ndcg_score([rels], preds)
    def mrr(l): 
        for i,v in enumerate(l,1):
            if v>0: return 1/i
        return 0
    st.subheader("📊 Evaluation")
    st.write(f"- NDCG@10: **{ndcg:.3f}**")
    st.write(f"- MRR: **{mrr(rels):.3f}**")
else:
    st.info("Like some results above to see evaluation metrics.")

# 🔟 Save user profile
profile_fn = "user_profile.json"
if not os.path.exists(profile_fn):
    with open(profile_fn,"w") as f:
        json.dump({"queries":[],"weights":{}, "liked":[]}, f)

with open(profile_fn,"r") as f:
    prof = json.load(f)

if st.button("💾 Save My Profile"):
    prof["queries"].append(raw_query)
    prof["weights"] = {
        "Size":w_size, "Citations":w_cite,
        "Diversity":w_div, "Recency":w_recency,
        "Similarity":w_sim
    }
    prof["liked"].extend(st.session_state.liked)
    prof["queries"] = list(dict.fromkeys(prof["queries"]))
    prof["liked"]   = list(dict.fromkeys(prof["liked"]))
    with open(profile_fn,"w") as f:
        json.dump(prof, f, indent=2)
    st.success("Profile saved!")

st.sidebar.header("👤 My Profile")
st.sidebar.write("• Queries:", prof.get("queries",[]))
st.sidebar.write("• Liked:", prof.get("liked",[]))
st.sidebar.write("• Weights:", prof.get("weights",{}))
