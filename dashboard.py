import streamlit as st
import pandas as pd
import os
import plotly.express as px
from joblib import load # Import joblib for loading models

# --- Configuration ---
st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Analyse de la Consommation d'Énergie dans le RAN 5G")
st.markdown("""
Bienvenue sur ce dashboard interactif permettant d'explorer la consommation
d'énergie en fonction des clusters issus du Deep Clustering appliqué à un DU simulé.
""")

st.title("🔍 Analyse Énergétique des Splits Fonctionnels dans le RAN 5G")

# --- Chargement des données et des modèles ---
st.sidebar.header("Configuration")

# Define file paths for data and models
data_file = "data/processed_data.csv"
# ae_model_path = "models/ae_deepcluster.h5" # COMMENTED OUT: Not using TensorFlow (.h5) as per request
cluster_model_path = "models/kmeans.joblib"
scaler_path = "models/scaler.joblib"

# Check if data file exists
if not os.path.exists(data_file):
    st.sidebar.error("Fichier de données non trouvé.")
    st.error(f"❌ Fichier introuvable : {data_file}")
    st.stop()

# Load main data
full_data = pd.read_csv(data_file)

# Load models using joblib
try:
    # If your autoencoder was also saved via joblib, you would load it here
    # For now, it's commented out as per "no tensorflow" and .h5 extension
    # ae_model = load(ae_model_path) 
    kmeans_model = load(cluster_model_path)
    scaler = load(scaler_path)
    st.sidebar.success("Modèles chargés avec succès !")
except FileNotFoundError:
    st.sidebar.error("Un ou plusieurs fichiers de modèle (.joblib) sont introuvables.")
    st.error("❌ Erreur : Vérifiez les chemins des fichiers modèles.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement des modèles : {e}")
    st.error(f"❌ Erreur critique : Impossible de charger les modèles. Détails : {e}")
    st.stop()


# --- Vérification des colonnes essentielles ---
# Ensure 'cluster_deep' column is present after loading data and potentially applying clustering
# (Assuming 'cluster_deep' is already in processed_data.csv or would be added by a K-Means model if applied directly here)
required_cols = ['cluster_deep', 'energy_per_packet', 'Split_Type']
missing = [c for c in required_cols if c not in full_data.columns]
if missing:
    st.error(f"❌ Colonnes essentielles manquantes dans 'processed_data.csv' : {', '.join(missing)}")
    st.stop()

# --- Aperçu des données ---
st.subheader("🔢 Aperçu des données")
st.dataframe(full_data.head(10))

# --- Visualisation : Répartition des clusters ---
st.subheader("📊 Répartition des clusters Deep Clustering")
cluster_counts = full_data['cluster_deep'].value_counts().sort_index()
fig1 = px.bar(
    cluster_counts.reset_index(), # Convert series to dataframe for px.bar
    x='index', y='cluster_deep', # Use 'index' for cluster labels, 'cluster_deep' for counts after value_counts()
    labels={'index': 'Cluster', 'cluster_deep': 'Nombre de points'}, # Update labels as per reset_index() output
    title="Nombre d’échantillons par cluster",
    color=cluster_counts.index.astype(str) # Ensure color is based on cluster index
)
st.plotly_chart(fig1, use_container_width=True)

# --- Visualisation : Énergie moyenne par cluster ---
st.subheader("⚡ Énergie moyenne par cluster Deep Clustering")
energy_mean = full_data.groupby('cluster_deep')['energy_per_packet'].mean().sort_values()
fig2 = px.bar(
    energy_mean.reset_index(), # Convert series to dataframe for px.bar
    x='cluster_deep', y='energy_per_packet', # Use actual column names from reset_index()
    labels={'cluster_deep': 'Cluster', 'energy_per_packet': 'Énergie moyenne (J/paquet)'},
    title="Consommation moyenne par cluster",
    color='energy_per_packet', # Color by energy value
    color_continuous_scale=px.colors.sequential.Viridis # Use a sequential scale for continuous values
)
st.plotly_chart(fig2, use_container_width=True)

# --- Visualisation : Distribution des types de split ---
st.subheader("📌 Distribution des splits dans chaque cluster")
split_dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
fig3 = px.bar(
    split_dist, barmode='stack',
    labels={'value': 'Proportion', 'cluster_deep': 'Cluster', 'Split_Type': 'Type de Split'},
    title="Proportion des types de splits par cluster",
    color_discrete_sequence=px.colors.qualitative.Pastel # Ensure colors are distinct for each Split_Type
)
st.plotly_chart(fig3, use_container_width=True)


# === Bouton interactif : Prédiction Split énergétique ===
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
