import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from joblib import load
import numpy as np

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
d'énergie en fonction des clusters issus du clustering Agglomerative.
""")

# --- Chargement des données et des modèles ---
st.sidebar.header("Configuration")

data_file = "data/processed_data.csv"
cluster_model_path = "models/agg.joblib"
scaler_path = "models/scaler.joblib"

if not os.path.exists(data_file):
    st.sidebar.error("Fichier de données non trouvé.")
    st.stop()

full_data = pd.read_csv(data_file)

try:
    agglo_model = load(cluster_model_path)
    scaler = load(scaler_path)
    st.sidebar.success("Modèles chargés avec succès ✅")
    
    # Récupérer les caractéristiques originales du scaler
    if hasattr(scaler, 'feature_names_in_'):
        feature_cols = scaler.feature_names_in_.tolist()
        st.sidebar.info(f"Caractéristiques du modèle: {len(feature_cols)} features")
    else:
        feature_cols = [
            'cpu_percent', 'cpu_freq', 'mem_usage',
            'net_sent', 'net_recv', 'energy_j',
            'cpu_deriv', 'mem_deriv', 'time_diff',
            'throughput_sent', 'throughput_recv',
            'tp_sent_roll_mean', 'tp_recv_roll_mean', 'tp_sent_roll_90pct',
            'delta_net_sent', 'delta_net_recv', 'delta_net_sum',
            'Split_Type', 'latence_classe'
        ]
        st.sidebar.warning("Utilisation des features par défaut - vérifiez la compatibilité")

except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# --- Vérification des colonnes ---
missing = [c for c in feature_cols if c not in full_data.columns]
if missing:
    st.error(f"❌ Colonnes manquantes dans le CSV pour le prétraitement : {missing}")
    st.stop()

# --- Prétraitement et prédiction des clusters ---
st.subheader("🔄 Prétraitement des données et prédiction des clusters")

# Extraction des features
X = full_data[feature_cols]

# Transformation en tableau numpy pour éviter la validation des noms
X_scaled = scaler.transform(X.to_numpy())

# Clustering Agglomerative
clusters = agglo_model.fit_predict(X_scaled)
full_data['cluster_agg'] = clusters

# --- Aperçu des données ---
st.subheader("🔢 Aperçu des données")
st.dataframe(full_data.head(10))

# --- Répartition des clusters ---
st.subheader("📊 Répartition des clusters Agglomerative")
cluster_counts = full_data['cluster_agg'].value_counts().sort_index()
cluster_df = cluster_counts.reset_index()
cluster_df.columns = ['cluster_agg', 'count']

fig1 = px.bar(
    cluster_df,
    x='cluster_agg', y='count',
    labels={'cluster_agg': 'Cluster', 'count': 'Nombre de points'},
    title="Nombre d'échantillons par cluster (Agglomerative)",
    color='cluster_agg'
)
st.plotly_chart(fig1, use_container_width=True)

# --- Énergie moyenne par cluster ---
st.subheader("⚡ Énergie moyenne par cluster Agglomerative")
if 'energy_per_packet' in full_data.columns:
    energy_mean = full_data.groupby('cluster_agg')['energy_per_packet'].mean().sort_values()
    fig2 = px.bar(
        energy_mean.reset_index(),
        x='cluster_agg', y='energy_per_packet',
        labels={'cluster_agg': 'Cluster', 'energy_per_packet': 'Énergie moyenne (J/paquet)'},
        title="Consommation moyenne par cluster (Agglomerative)",
        color='energy_per_packet',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("La colonne 'energy_per_packet' est manquante pour l'analyse énergétique")

# --- Distribution des splits par cluster ---
st.subheader("📌 Distribution des splits dans chaque cluster")
if 'Split_Type' in full_data.columns:
    split_dist = (
        full_data
        .groupby('cluster_agg')['Split_Type']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    fig3 = px.bar(
        split_dist,
        barmode='stack',
        labels={'value':'Proportion','cluster_agg':'Cluster','Split_Type':'Type de Split'},
        title="Proportion des types de splits par cluster (Agglomerative)",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("La colonne 'Split_Type' est manquante pour l'analyse des splits")

# --- Analyse interactive des splits ---
st.subheader("🔍 Analyse personnalisée des splits")
if 'Split_Type' in full_data.columns and 'energy_per_packet' in full_data.columns:
    choice = st.radio(
        "Souhaitez-vous identifier :",
        ["Le split le PLUS énergivore", "Le split le MOINS énergivore"]
    )

    split_energy = full_data.groupby('Split_Type')['energy_per_packet'].mean()

    if choice == "Le split le PLUS énergivore":
        s = split_energy.idxmax()
        v = split_energy.max()
        st.success(f"🔺 Le split **{int(s)}** est le PLUS énergivore avec **{v:.2e} J/paquet**.")
    else:
        s = split_energy.idxmin()
        v = split_energy.min()
        st.success(f"🔻 Le split **{int(s)}** est le MOINS énergivore avec **{v:.2e} J/paquet**.")
else:
    st.warning("Colonnes manquantes pour l'analyse des splits")

# --- Bouton de rafraîchissement ---
if st.button("🔄 Rafraîchir les données"):
    st.experimental_rerun()
