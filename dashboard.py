import streamlit as st
import pandas as pd
import os
import plotly.express as px
from joblib import load

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Dashboard Deep Clustering 5G", layout="wide")
st.markdown("""
    <style>
      body { background-color: #f5f7fa; }
      h1, h2 { color: #003865; }
      .stButton>button { background-color: #1976d2; color: white; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

st.title("📡 Dashboard Deep Clustering – RAN 5G")
st.markdown("Analyse énergétique et segmentation des comportements DU via Deep Clustering")

# --- CHEMINS ---
data_file = "data/processed_data.csv"
scaler_path = "models/scaler.joblib"
encoder_path = "models/encoder.joblib"
cluster_model_path = "models/cluster_deep.joblib"

# --- VERIFICATION ---
for path in [data_file, scaler_path, encoder_path, cluster_model_path]:
    if not os.path.exists(path):
        st.error(f"❌ Fichier introuvable : {path}")
        st.stop()

# --- CHARGEMENT ---
full_data = pd.read_csv(data_file)
scaler = load(scaler_path)
encoder = load(encoder_path)
cluster_model = load(cluster_model_path)

st.success("✅ Données et modèles chargés")

# --- COLONNES DE FEATURES ---
feature_cols = [
    'cpu_percent', 'cpu_freq', 'mem_usage', 'net_sent', 'net_recv',
    'cpu_deriv', 'mem_deriv', 'time_diff', 'throughput_sent', 'throughput_recv',
    'tp_sent_roll_mean', 'tp_recv_roll_mean', 'tp_sent_roll_90pct',
    'delta_net_sent', 'delta_net_recv', 'delta_net_sum', 'energy_j',
    'Split_Type', 'latence_classe', 'energy_per_packet'
]

# --- BOUTON DE PREDICTION ---
if st.button("🔍 Prédire les clusters Deep Clustering"):
    try:
        # --- SELECTION + NORMALISATION ---
        X = full_data[feature_cols]
        X_scaled = scaler.transform(X)

        # --- ENCODAGE LATENT ---
        latent_features = encoder.transform(X_scaled)

        # --- CLUSTERING ---
        clusters = cluster_model.predict(latent_features)
        full_data['cluster_deep'] = clusters

        st.success("✅ Clusters prédits et ajoutés au DataFrame")

        # --- VISUALISATION ---
        st.subheader("📊 Répartition des clusters")
        cluster_counts = full_data['cluster_deep'].value_counts().sort_index()
        fig1 = px.bar(
            cluster_counts.reset_index(),
            x='index', y='cluster_deep',
            labels={'index': 'Cluster', 'cluster_deep': 'Nombre de points'},
            title="Nombre d’échantillons par cluster",
            color=cluster_counts.index.astype(str)
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("⚡ Énergie moyenne par cluster")
        energy_mean = full_data.groupby('cluster_deep')['energy_per_packet'].mean().sort_values()
        fig2 = px.bar(
            energy_mean.reset_index(),
            x='cluster_deep', y='energy_per_packet',
            labels={'cluster_deep': 'Cluster', 'energy_per_packet': 'Énergie moyenne (J/paquet)'},
            title="Énergie moyenne par cluster",
            color='energy_per_packet', color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📌 Distribution des types de split par cluster")
        split_dist = full_data.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
        fig3 = px.bar(
            split_dist,
            barmode='stack',
            labels={'value': 'Proportion', 'cluster_deep': 'Cluster', 'Split_Type': 'Type de Split'},
            title="Proportion des types de splits par cluster",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig3, use_container_width=True)

        # --- Analyse split interactif ---
        st.subheader("🔎 Analyse personnalisée des splits")
        choice = st.radio("Souhaitez-vous identifier :", ["Le split le PLUS énergivore", "Le split le MOINS énergivore"])

        split_energy = full_data.groupby('Split_Type')['energy_per_packet'].mean()

        if choice == "Le split le PLUS énergivore":
            target_split = split_energy.idxmax()
            value = split_energy.max()
            st.error(f"⚡ Le split **{int(target_split)}** est le PLUS énergivore avec {value:.2e} J/paquet.")
        else:
            target_split = split_energy.idxmin()
            value = split_energy.min()
            st.success(f"✅ Le split **{int(target_split)}** est le MOINS énergivore avec {value:.2e} J/paquet.")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

else:
    st.info("Cliquez sur le bouton ci-dessus pour générer et afficher les clusters.")
