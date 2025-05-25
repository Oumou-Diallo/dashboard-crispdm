import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.title("Dashboard de Visualisation📊")

# 1. Chargement des données pré-calculées
data_file = "data/processed_data.csv"
if not os.path.exists(data_file):
    st.error(f"Le fichier {data_file} est introuvable.")
    st.stop()
full_data = pd.read_csv(data_file)

# 2. Vérifier la présence de la colonne 'cluster_deep'
if 'cluster_deep' not in full_data.columns:
    st.error("La colonne 'cluster_deep' est introuvable dans les données. Assurez-vous d'avoir exporté full_data.csv avec cette colonne.")
    st.stop()

# 3. Affichage interactif
st.subheader("Aperçu des données avec clusters deep")
st.dataframe(full_data.head())

# 4. Statistiques et visualisations
# 4.1 Répartition des clusters deep
st.subheader("Répartition des clusters Deep Clustering")
cluster_counts = full_data['cluster_deep'].value_counts().sort_index()
st.bar_chart(cluster_counts)

# 4.2 Énergie moyenne par cluster deep
if 'energy_per_packet' in full_data.columns:
    st.subheader("Énergie moyenne par cluster Deep Clustering")
    energy_mean = full_data.groupby('cluster_deep')['energy_per_packet'].mean()
    st.bar_chart(energy_mean)
else:
    st.warning("La colonne 'energy_per_packet' est manquante.")

# 4.3 Distribution des types de split par cluster deep
if 'Split_Type' in full_data.columns:
    st.subheader("Distribution des types de split par cluster Deep Clustering")
    dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
    st.dataframe(dist)
else:
    st.warning("La colonne 'Split_Type' est manquante.")
