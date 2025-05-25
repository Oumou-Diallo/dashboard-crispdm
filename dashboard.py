import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.title("Consommation D'énergie Dans les Réseaux D'accès📊")

# 1. Chargement des données pré-calculées
data_file = "data/processed_data.csv"
if not os.path.exists(data_file):
    st.error(f"Le fichier {data_file} est introuvable.")
    st.stop()
full_data = pd.read_csv(data_file)

# 2. Vérifier la présence des colonnes nécessaires
required_cols = ['cluster_deep', 'energy_per_packet', 'Split_Type']
missing = [c for c in required_cols if c not in full_data.columns]
if missing:
    st.error(f"Colonnes manquantes pour l'affichage: {', '.join(missing)}")
    st.stop()

# 3. Affichage interactif
st.subheader("Aperçu des données avec clusters deep")
st.dataframe(full_data.head())

# 4. Statistiques et visualisations
# 4.1 Répartition des clusters Deep Clustering
st.subheader("Répartition des clusters Deep Clustering")
cluster_counts = full_data['cluster_deep'].value_counts().sort_index()
st.bar_chart(cluster_counts)

# 4.2 Énergie moyenne par cluster Deep Clustering
st.subheader("Énergie moyenne par cluster Deep Clustering")
energy_mean = full_data.groupby('cluster_deep')['energy_per_packet'].mean()
st.bar_chart(energy_mean)

# 4.3 Distribution des types de split par cluster Deep Clustering
st.subheader("Distribution des types de split par cluster Deep Clustering")
dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
st.dataframe(dist)

# 5. Quel split consomme le plus d'énergie ?
st.subheader("Split le plus énergivore")
split_energy = full_data.groupby('Split_Type')['energy_per_packet'].mean()
worst_split = split_energy.idxmax()
worst_value = split_energy.max()
st.write(f"Le type de split **{int(worst_split)}** consomme le plus avec en moyenne **{worst_value:.2e}** joules par paquet.")
