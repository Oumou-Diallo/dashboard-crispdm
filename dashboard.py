import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configuration
st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.title("🔍 Analyse Énergétique des Splits Fonctionnels dans le RAN 5G")

# === Chargement des données ===
data_file = "data/processed_data.csv"
if not os.path.exists(data_file):
    st.error("❌ Fichier introuvable : data/processed_data.csv")
    st.stop()

full_data = pd.read_csv(data_file)

# === Vérification des colonnes essentielles ===
required_cols = ['cluster_deep', 'energy_per_packet', 'Split_Type']
missing = [c for c in required_cols if c not in full_data.columns]
if missing:
    st.error(f"Colonnes manquantes pour l’affichage : {', '.join(missing)}")
    st.stop()

# === Aperçu des données ===
st.subheader("🧾 Aperçu des données")
st.dataframe(full_data.head(10))

# === Visualisation : Répartition des clusters ===
st.subheader("📊 Répartition des clusters Deep Clustering")
cluster_counts = full_data['cluster_deep'].value_counts().sort_index()
fig1 = px.bar(cluster_counts, labels={'index': 'Cluster', 'value': 'Nombre de points'},
              title="Nombre d’échantillons par cluster", color=cluster_counts.index)
st.plotly_chart(fig1, use_container_width=True)

# === Visualisation : Énergie moyenne par cluster ===
st.subheader("⚡ Énergie moyenne par cluster Deep Clustering")
energy_mean = full_data.groupby('cluster_deep')['energy_per_packet'].mean().sort_values()
fig2 = px.bar(energy_mean, labels={'index': 'Cluster', 'value': 'Énergie moyenne (J/paquet)'},
              title="Consommation moyenne par cluster", color=energy_mean.index)
st.plotly_chart(fig2, use_container_width=True)

# === Visualisation : Split_Type par cluster ===
st.subheader("📌 Distribution des splits dans chaque cluster")
split_dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
st.dataframe(split_dist.style.format("{:.2%}"))

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
