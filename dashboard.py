import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Dashboard CRISP-DM", layout="wide")
st.title("Dashboard de visualisation sans TensorFlow")

# 1. Chargement des données pré-traitées
data_file = "data/processed_data.csv"
if not os.path.exists(data_file):
    st.error(f"Le fichier {data_file} est introuvable.")
    st.stop()
full_data = pd.read_csv(data_file)

# 2. Définir la liste des colonnes utilisées pour prédiction
# Doivent correspondre exactement à celles utilisées lors de l'entraînement
feature_cols = [
    'cpu_percent', 'cpu_freq', 'mem_usage', 'net_sent', 'net_recv',
    'cpu_deriv', 'mem_deriv', 'time_diff', 'throughput_sent', 'throughput_recv',
    'tp_sent_roll_mean', 'tp_recv_roll_mean', 'tp_sent_roll_90pct',
    'delta_net_sent', 'delta_net_recv', 'delta_net_sum', 'energy_j',
    'Split_Type', 'latence_classe','energy_per_packet'
    # plus toutes les colonnes de one-hot pour Split_Type
] 

# 3. Vérification des artefacts de scaling et clustering
scaler_path = "models/scaler.joblib"
cluster_model_path = "models/cluster_deep.joblib"  # ou ton modèle final joblib

if not os.path.exists(scaler_path) or not os.path.exists(cluster_model_path):
    st.warning("Le scaler ou le modèle de clustering est manquant dans 'models/'.")
    st.stop()

scaler = joblib.load(scaler_path)
cluster_model = joblib.load(cluster_model_path)
st.success("Scaler et modèle de clustering chargés avec succès.")

# 4. Prédiction interactive
if st.button("Prédire les clusters deep" ):
    try:
        # Sélection et mise à l'échelle
        X = full_data[feature_cols]
        X_scaled = scaler.transform(X)

        # Prédiction des clusters
        clusters = cluster_model.predict(X_scaled)
        full_data['Cluster'] = clusters

        # 5. Affichage des résultats
        st.subheader("Aperçu des prédictions Deep Clustering")
        st.dataframe(full_data[['Cluster'] + feature_cols].head())

        # Graphiques
        st.subheader("Répartition des clusters")
        st.bar_chart(full_data['Cluster'].value_counts().sort_index())

        st.subheader("Énergie moyenne par cluster")
        energy_mean = full_data.groupby('Cluster')['energy_per_packet'].mean()
        st.bar_chart(energy_mean)
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
