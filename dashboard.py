import streamlit as st
import pandas as pd
import os
import plotly.express as px

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

# Chargement des données
st.sidebar.header("Configuration")
data_file = "data/processed_data.csv"
if not os.path.exists(data_file):
    st.sidebar.error("Fichier de données non trouvé.")
    st.stop()

full_data = pd.read_csv(data_file)

# Vérification
required_cols = ['cluster_deep', 'energy_per_packet', 'Split_Type']
missing = [c for c in required_cols if c not in full_data.columns]
if missing:
    st.error(f"Colonnes manquantes : {', '.join(missing)}")
    st.stop()

# Aperçu
st.subheader("🔢 Aperçu des données")
st.dataframe(full_data.head(), height=200)

# Répartition des clusters
st.subheader("📊 Répartition des Clusters")
fig1 = px.bar(
    full_data['cluster_deep'].value_counts().sort_index().reset_index(),
    x='index', y='cluster_deep',
    labels={'index': 'Cluster', 'cluster_deep': 'Nombre de points'},
    color_discrete_sequence=['#636EFA']
)
st.plotly_chart(fig1, use_container_width=True)

# Énergie moyenne par cluster
st.subheader("🔋 Énergie Moyenne par Cluster")
fig2 = px.bar(
    full_data.groupby('cluster_deep')['energy_per_packet'].mean().reset_index(),
    x='cluster_deep', y='energy_per_packet',
    labels={'cluster_deep': 'Cluster', 'energy_per_packet': 'Énergie moyenne (J/paquet)'},
    color='energy_per_packet', color_continuous_scale='Viridis'
)
st.plotly_chart(fig2, use_container_width=True)

# Distribution des types de split
st.subheader("🔹 Distribution des Splits par Cluster")
dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
fig3 = px.bar(
    dist, barmode='stack',
    labels={'value': 'Proportion'},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig3, use_container_width=True)

# Split le plus énergivore
st.subheader("⚡ Split le Plus Énergivore")
split_energy = full_data.groupby('Split_Type')['energy_per_packet'].mean()
worst_split = split_energy.idxmax()
worst_value = split_energy.max()
st.info(f"Le Functional Split **{int(worst_split)}** est le plus énergivore avec **{worst_value:.2e} J/paquet**")
