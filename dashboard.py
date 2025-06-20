import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import tensorflow as tf

st.set_page_config(page_title="Dashboard Deep Clustering", layout="wide")
st.title("📡 Analyse Énergétique via Deep Clustering dans le RAN 5G")

# === 1. Chargement des données ===
data_path = "data/processed_data.csv"
if not os.path.exists(data_path):
    st.error("❌ Fichier introuvable : data/processed_data.csv")
    st.stop()

df = pd.read_csv(data_path)

# === 2. Chargement des artefacts ===
ae_model_path = "models/ae_deepcluster.h5"
cluster_model_path = "models/kmeans.joblib"
scaler_path = "models/scaler.joblib"

if not all(os.path.exists(p) for p in [ae_model_path, cluster_model_path, scaler_path]):
    st.error("❌ Un ou plusieurs fichiers de modèle sont manquants dans le dossier 'models/'.")
    st.stop()

autoencoder = tf.keras.models.load_model(ae_model_path)
cluster_model = joblib.load(cluster_model_path)
scaler = joblib.load(scaler_path)

st.success("✅ Modèles (Autoencodeur + KMeans + Scaler) chargés avec succès.")

# === 3. Prédiction des clusters ===
try:
    X = df.select_dtypes(include='number')  # Sélection simple
    X_scaled = scaler.transform(X)
    features_latents = autoencoder.predict(X_scaled)
    clusters = cluster_model.predict(features_latents)
    df['cluster_deep'] = clusters
    st.success("✅ Clusters prédits avec succès.")
except Exception as e:
    st.error(f"Erreur pendant la prédiction : {e}")
    st.stop()

# === 4. Aperçu des données ===
st.subheader("🧾 Données avec les clusters")
st.dataframe(df.head(10))

# === 5. Visualisation des résultats ===
st.subheader("📊 Répartition des clusters Deep Clustering")
fig1 = px.bar(df['cluster_deep'].value_counts().sort_index(),
              labels={"index": "Cluster", "value": "Nombre d'échantillons"},
              title="Nombre d’échantillons par cluster",
              color=df['cluster_deep'].value_counts().sort_index().index)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("⚡ Énergie moyenne par cluster")
if 'energy_per_packet' in df.columns:
    fig2 = px.bar(df.groupby('cluster_deep')['energy_per_packet'].mean(),
                  labels={"index": "Cluster", "value": "Énergie moyenne (J/paquet)"},
                  title="Consommation moyenne d'énergie",
                  color=df.groupby('cluster_deep')['energy_per_packet'].mean().index)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Colonne 'energy_per_packet' manquante pour calculer l’énergie.")

# === 6. Split par cluster (si dispo) ===
if 'Split_Type' in df.columns:
    st.subheader("📌 Distribution des Splits par Cluster")
    split_dist = df.groupby('cluster_deep')['Split_Type'].value_counts(normalize=True).unstack(fill_value=0)
    st.dataframe(split_dist.style.format("{:.2%}"))
else:
    st.info("Colonne 'Split_Type' non disponible pour affichage.")

# === 7. Bouton interactif : split le plus ou le moins énergivore ===
st.subheader("🔍 Analyse personnalisée des splits")
if 'Split_Type' in df.columns and 'energy_per_packet' in df.columns:
    choix = st.radio("Souhaitez-vous identifier :", ["Le split le PLUS énergivore", "Le split le MOINS énergivore"])
    split_energy = df.groupby('Split_Type')['energy_per_packet'].mean()

    if choix == "Le split le PLUS énergivore":
        worst = split_energy.idxmax()
        val = split_energy.max()
        st.error(f"🔺 Le split **{int(worst)}** est le PLUS énergivore avec **{val:.2e} J/paquet**.")
    else:
        best = split_energy.idxmin()
        val = split_energy.min()
        st.success(f"🔻 Le split **{int(best)}** est le MOINS énergivore avec **{val:.2e} J/paquet**.")
else:
    st.warning("⚠️ Analyse impossible : 'Split_Type' ou 'energy_per_packet' manquant.")
