import streamlit as st
import pandas as pd
import os
import plotly.express as px
from joblib import load

# --- Configuration générale ---
st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
      background-color: #4CAF50; color: white; border-radius: 8px;
      padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Analyse de la Consommation d'Énergie dans le RAN 5G")
st.markdown("""
Explorer la consommation énergétique selon les clusters générés 
par Agglomerative Clustering.
""")

# --- Chargement des données et des modèles ---
st.sidebar.header("Configuration")

DATA_PATH     = "data/processed_data.csv"
MODEL_PATH    = "models/agg.joblib"
SCALER_PATH   = "models/scaler.joblib"

if not os.path.exists(DATA_PATH):
    st.sidebar.error("❌ processed_data.csv introuvable")
    st.stop()

df = pd.read_csv(DATA_PATH)

try:
    agglo = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    st.sidebar.success("Modèles chargés ✅")
except Exception as e:
    st.sidebar.error(f"Erreur chargement modèles : {e}")
    st.stop()

# --- Liste FIXE des colonnes utilisées pour scaler et clustering ---
feature_cols = [
    'cpu_percent','cpu_freq','mem_usage',
    'net_sent','net_recv','energy_j',
    'cpu_deriv','mem_deriv','time_diff',
    'throughput_sent','throughput_recv',
    'tp_sent_roll_mean','tp_recv_roll_mean','tp_sent_roll_90pct',
    'delta_net_sent','delta_net_recv','delta_net_sum',
    'Split_Type','latence_classe'
]

# Vérification
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"❌ Colonnes manquantes pour le scaler : {missing}")
    st.stop()

# --- Prétraitement & clustering ---
st.subheader("🔄 Prétraitement & Clustering")

X = df[feature_cols]
X_scaled = scaler.transform(X)
clusters = agglo.fit_predict(X_scaled)
df['cluster_deep'] = clusters

# --- Affichages ---
st.subheader("🔢 Données (aperçu)")
st.dataframe(df.head(10))

# 1) Répartition des clusters
st.subheader("📊 Répartition des clusters")
counts = df['cluster_deep'].value_counts().sort_index().reset_index()
counts.columns = ['cluster','count']
fig1 = px.bar(counts, x='cluster', y='count',
              title="Nombre d'échantillons par cluster",
              labels={'cluster':'Cluster','count':'Nb points'})
st.plotly_chart(fig1, use_container_width=True)

# 2) Énergie moyenne par cluster
st.subheader("⚡ Énergie moyenne par cluster")
em = df.groupby('cluster_deep')['energy_per_packet'].mean().reset_index()
em.columns = ['cluster','energy_mean']
fig2 = px.bar(em, x='cluster', y='energy_mean',
              title="Énergie moyenne (J/paquet)", color='energy_mean',
              color_continuous_scale='Viridis')
st.plotly_chart(fig2, use_container_width=True)

# 3) Distribution des splits
st.subheader("📌 Répartition des splits par cluster")
dist = (df.groupby('cluster_deep')['Split_Type']
          .value_counts(normalize=True)
          .unstack(fill_value=0))
fig3 = px.bar(dist, barmode='stack',
              title="Proportion des types de split",
              labels={'value':'Proportion','cluster_deep':'Cluster'})
st.plotly_chart(fig3, use_container_width=True)

# 4) Analyse interactive
st.subheader("🔍 Quel split ?")
choice = st.radio("Voir :", ["Le plus énergivore","Le moins énergivore"])
se = df.groupby('Split_Type')['energy_per_packet'].mean()
if choice=="Le plus énergivore":
    s,v = se.idxmax(),se.max()
    st.success(f"🔺 Split {int(s)} (≈{v:.2e} J/paquet)")
else:
    s,v = se.idxmin(),se.min()
    st.success(f"🔻 Split {int(s)} (≈{v:.2e} J/paquet)")

# 5) Rafraîchir
if st.button("🔄 Rafraîchir"):
    st.experimental_rerun()
