import streamlit as st
import pandas as pd
import joblib

st.title("Dashboard de Clustering")

# Charger données ou résultats
df = pd.read_csv("data/mon_fichier.csv")
model = joblib.load("models/kmeans.joblib")

st.write("Aperçu des données")
st.dataframe(df.head())
