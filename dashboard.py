# dashboard.py
# dashboard.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.title("Dashboard de visualisation avec Deep Clustering")

# Chargement des données
data_file = "data/data.csv"
if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    st.write("Aperçu des données :")
    st.dataframe(df.head())
else:
    st.error(f"Le fichier {data_file} est introuvable.")

# Vérification si les fichiers nécessaires existent
ae_model_path = "models/ae_deepcluster.h5"
cluster_model_path = "models/kmeans.joblib"
scaler_path = "models/scaler.joblib"

if all(os.path.exists(path) for path in [ae_model_path, cluster_model_path, scaler_path]):
    # Chargement des modèles
    autoencoder = tf.keras.models.load_model(ae_model_path)
    cluster_model = joblib.load(cluster_model_path)
    scaler = joblib.load(scaler_path)
    
    st.success("Modèle autoencodeur et modèle de clustering chargés.")

    if st.button("Faire des prédictions Deep Cluster"):
        try:
            # Préparation des données numériques
            num_data = df.select_dtypes(include="number")
            num_data_scaled = scaler.transform(num_data)
            
            # Extraction des features latentes
            latent_features = autoencoder.predict(num_data_scaled)

            # Clustering sur les features latentes
            cluster_preds = cluster_model.predict(latent_features)
            df["Cluster"] = cluster_preds

            st.subheader("Données avec les clusters détectés :")
            st.dataframe(df[["Cluster"] + list(num_data.columns)])
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.warning("Un ou plusieurs fichiers de modèle sont manquants dans le dossier 'models'.")
