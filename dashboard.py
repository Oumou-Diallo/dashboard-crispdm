import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignorer les avertissements de version
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from joblib import load
import numpy as np
import h5py  # Pour charger les modèles HDF5

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
d'énergie en fonction des clusters issus du Deep Clustering appliqué à un DU simulé.
""")

# --- Chargement des données et des modèles ---
st.sidebar.header("Configuration")

data_file = "data/processed_data.csv"
cluster_model_path = "models/kmeans.joblib"
scaler_path = "models/scaler.joblib"
autoencoder_path = "models/ae_deepcluster.h5"
features_path = "models/features.joblib"

if not os.path.exists(data_file):
    st.sidebar.error("Fichier de données non trouvé.")
    st.stop()

full_data = pd.read_csv(data_file)

try:
    # Charger tous les modèles nécessaires
    kmeans_model = load(cluster_model_path)
    scaler = load(scaler_path)
    
    # Charger la liste des features
    feature_columns = load(features_path)
    
    # Charger l'autoencodeur sans TensorFlow
    def load_autoencoder_model(path):
        with h5py.File(path, 'r') as f:
            weights = [np.array(f[f'weight_{i}']) for i in range(len(f.keys()))]
        return weights
    
    autoencoder_weights = load_autoencoder_model(autoencoder_path)
    
    st.sidebar.success("Modèles chargés avec succès ✅")
    
    # Afficher les features chargées (optionnel)
    st.sidebar.subheader("Features utilisées")
    st.sidebar.write(f"{len(feature_columns)} colonnes chargées")
    if st.sidebar.checkbox("Afficher la liste complète"):
        st.sidebar.write(feature_columns)
        
except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# --- Vérification des colonnes ---
missing = [c for c in feature_columns if c not in full_data.columns]
if missing:
    st.error(f"❌ Colonnes manquantes dans les données : {', '.join(missing)}")
    st.stop()

# --- Calcul des clusters en temps réel ---
@st.cache_data
def predict_with_autoencoder(X, weights):
    """Fonction de prédiction manuelle pour l'autoencodeur"""
    # Cette fonction doit être adaptée à votre architecture d'autoencodeur
    # Voici un exemple simplifié pour une architecture dense simple
    num_layers = len(weights) // 2
    current_layer = X
    
    for i in range(num_layers):
        w = weights[2*i]
        b = weights[2*i + 1]
        current_layer = np.dot(current_layer, w) + b
        if i < num_layers - 1:  # Pas d'activation pour la dernière couche
            current_layer = np.maximum(0, current_layer)  # ReLU
        
    return current_layer

@st.cache_data
def calculate_clusters(data, _autoencoder_weights, _kmeans, _scaler, features):
    """Calcule les clusters à partir des données brutes"""
    # Sélection et normalisation des features
    X = data[features]
    X_scaled = _scaler.transform(X)
    
    # Représentation latente via Autoencoder
    latent_rep = predict_with_autoencoder(X_scaled, _autoencoder_weights)
    
    # Prédiction des clusters avec KMeans
    clusters = _kmeans.predict(latent_rep)
    return clusters

# Ajout des clusters aux données
full_data['cluster_deep'] = calculate_clusters(
    full_data, 
    autoencoder_weights, 
    kmeans_model, 
    scaler, 
    feature_columns
)

# --- Le reste du code reste inchangé à partir d'ici ---
# ... [Le code des visualisations que vous aviez précédemment] ...
