import streamlit as st
import pandas as pd
import os
import plotly.express as px
from joblib import load  # Import joblib for loading models

# --- Configuration générale ---
st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Analyse de la Consommation d'Énergie dans le RAN 5G")
st.markdown("""
Bienvenue sur ce dashboard interactif permettant d'explorer la consommation
d'énergie en fonction des clusters issus du Deep Clustering appliqué à un DU simulé.
""")

# --- Chargement des données et des modèles ---
st.sidebar.header("Configuration")

data_file = "data/processed_data.csv"
cluster_model_path = "models/kmeans.joblib"
scaler_path = "models/scaler.joblib"

if not os.path.exists(data_file):
    st.sidebar.error("Fichier de données non trouvé.")
    st.stop()

full_data = pd.read_csv(data_file)

try:
    kmeans_model = load(cluster_model_path)
    scaler = load(scaler_path)
    st.sidebar.success("Modèles chargés avec succès ✅")
except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# --- Vérification des colonnes ---
required_cols = ['cluster_deep', 'energy_per_packet', 'Split_Type']
missing = [c for c in required_cols if c not in full_data.columns]
if missing:
    st.error(f"❌ Colonnes manquantes dans 'processed_data.csv' : {', '.join(missing)}")
    st.stop()

# --- Aperçu des données ---
st.subheader("🔢 Aperçu des données")
st.dataframe(full_data.head(10))

# --- Répartition des clusters ---
st.subheader("📊 Répartition des clusters Deep Clustering")
cluster_counts = full_data['cluster_deep'].value_counts().sort_index()
cluster_df = cluster_counts.reset_index()
cluster_df.columns = ['cluster_deep', 'count']

fig1 = px.bar(
    cluster_df,
    x='cluster_deep', y='count',
    labels={'cluster_deep': 'Cluster', 'count': 'Nombre de points'},
    title="Nombre d’échantillons par cluster",
    color='cluster_deep'
)
st.plotly_chart(fig1, use_container_width=True)

# --- Énergie moyenne par cluster ---
st.subheader("⚡ Énergie moyenne par cluster Deep Clustering")
energy_mean = full_data.groupby('cluster_deep')['energy_per_packet'].mean().sort_values()
fig2 = px.bar(
    energy_mean.reset_index(),
    x='cluster_deep', y='energy_per_packet',
    labels={'cluster_deep': 'Cluster', 'energy_per_packet': 'Énergie moyenne (J/paquet)'},
    title="Consommation moyenne par cluster",
    color='energy_per_packet',
    color_continuous_scale=px.colors.sequential.Viridis
)
st.plotly_chart(fig2, use_container_width=True)

# --- Distribution des splits par cluster ---
st.subheader("📌 Distribution des splits dans chaque cluster")
split_dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
fig3 = px.bar(
    split_dist,
    barmode='stack',
    labels={'value': 'Proportion', 'cluster_deep': 'Cluster', 'Split_Type': 'Type de Split'},
    title="Proportion des types de splits par cluster",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig3, use_container_width=True)

# --- Analyse interactive des splits ---
st.subheader("🔍 Analyse personnalisée des splits")
choice = st.radio("Souhaitez-vous identifier :", ["Le split le PLUS énergivore", "Le split le MOINS énergivore"])

split_energy = full_data.groupby('Split_Type')['energy_per_packet'].mean()

if choice == "Le split le PLUS énergivore":
    target_split = split_energy.idxmax()
    value = split_energy.max()
    st.success(f"🔺 Le split **{int(target_split)}** est le PLUS énergivore avec une moyenne de **{value:.2e} J/paquet**.")
else:
    target_split = split_energy.idxmin()
    value = split_energy.min()
    st.success(f"🔻 Le split **{int(target_split)}** est le MOINS énergivore avec une moyenne de **{value:.2e} J/paquet**.")

# --- Option bouton de rechargement / reset ---
if st.button("🔄 Rafraîchir les données"):
    st.experimental_rerun()
