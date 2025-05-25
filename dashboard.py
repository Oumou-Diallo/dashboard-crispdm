import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.title("Dashboard de visualisation sans TensorFlow")

# 1. Chargement des données pré-traitées
data_file = "data/processed_data.csv"
if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    st.write("Aperçu des données :")
    st.dataframe(df.head())
else:
    st.error(f"Le fichier {data_file} est introuvable.")
    st.stop()

# 2. Vérification du scaler et du modèle de clustering
cluster_model_path = "models/kmeans.joblib"  # ou kmeans.joblib si tu préfères
scaler_path = "models/scaler.joblib"

if not os.path.exists(cluster_model_path) or not os.path.exists(scaler_path):
    st.warning("Le modèle de clustering ou le scaler est manquant dans 'models/'.")
    st.stop()

# 3. Chargement des artefacts
scaler = joblib.load(scaler_path)
cluster_model = joblib.load(cluster_model_path)
st.success("Scaler et modèle de clustering chargés avec succès.")

# 4. Bouton de prédiction
if st.button("Prédire les clusters"):
    try:
        # On sélectionne uniquement les colonnes numériques pour scaler
        num_cols = df.select_dtypes(include="number").columns.tolist()
        X = df[num_cols]
        X_scaled = scaler.transform(X)

        # Prédiction des clusters
        clusters = cluster_model.predict(X_scaled)
        df["Cluster"] = clusters

        # Affichage des résultats
        st.subheader("Résultats de clustering")
        st.dataframe(df[["Cluster"] + num_cols])

        # Visualisations
        st.subheader("Taille des clusters")
        st.bar_chart(df["Cluster"].value_counts().sort_index())

        st.subheader("Énergie moyenne par cluster")
        energy_mean = df.groupby("Cluster")["energy_per_packet"].mean()
        st.bar_chart(energy_mean)

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
