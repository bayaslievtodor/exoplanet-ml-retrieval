import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import ExoplanetCNN

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
CSV_PATH = os.path.join(BASE_DIR, 'demo_planets.csv')

CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'predictor_model_cnn.pt')
SPEC_SCALER_PATH = os.path.join(MODEL_DIR, 'spec_scaler.pkl')
AUX_SCALER_PATH = os.path.join(MODEL_DIR, 'aux_scaler.pkl')

GASES = ['H₂O', 'CO₂', 'CO', 'CH₄', 'NH₃']
SPECTRAL_COLS = [f'ch_{i+1}' for i in range(52)]
AUX_COLS = [
    'star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature',
    'planet_mass_kg', 'planet_orbital_period', 'planet_distance',
    'planet_surface_gravity'
]
TARGET_COLS = [
    'log_H2O_q1', 'log_H2O_q2', 'log_H2O_q3',
    'log_CO2_q1', 'log_CO2_q2', 'log_CO2_q3',
    'log_CO_q1',  'log_CO_q2',  'log_CO_q3',
    'log_CH4_q1', 'log_CH4_q2', 'log_CH4_q3',
    'log_NH3_q1', 'log_NH3_q2', 'log_NH3_q3'
]

@st.cache_resource
def load_assets():
    cnn = ExoplanetCNN()
    cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
    cnn.eval()
    spec_scaler = joblib.load(SPEC_SCALER_PATH)
    aux_scaler = joblib.load(AUX_SCALER_PATH)
    return cnn, spec_scaler, aux_scaler

cnn_model, spec_scaler, aux_scaler = load_assets()
df = pd.read_csv(CSV_PATH, index_col='planet_ID')

available_ids = df.index.tolist()[:5]
greek_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
planet_mapping = dict(zip(greek_names, available_ids))

st.title("Exoplanet Atmospheric Retrieval Evaluation")
st.markdown("---")

selected_nickname = st.selectbox("Select Target Planet", greek_names)
target_id = planet_mapping[selected_nickname]

row = df.loc[target_id]
spec_raw = row[SPECTRAL_COLS].values.astype(float).reshape(1, -1)
aux_raw = row[AUX_COLS].values.astype(float).reshape(1, -1)
true_ret = row[TARGET_COLS].values.astype(float)

spec_scaled = spec_scaler.transform(spec_raw)
aux_scaled = aux_scaler.transform(aux_raw)
spec_t = torch.tensor(spec_scaled, dtype=torch.float32).unsqueeze(1)
aux_t = torch.tensor(aux_scaled, dtype=torch.float32)

with torch.no_grad():
    cnn_pred = cnn_model(spec_t, aux_t).numpy().flatten()

meta_cols = st.columns(4)
with meta_cols[0]:
    st.metric("Surface Gravity", f"{row['planet_surface_gravity']:.2f} m/s²")
with meta_cols[1]:
    st.metric("Orbital Period", f"{row['planet_orbital_period']:.2f} Days")
with meta_cols[2]:
    st.metric("Stellar Temperature", f"{int(row['star_temperature'])} K")
with meta_cols[3]:
    st.metric("Distance", f"{row['planet_distance']:.3f} AU")

st.markdown("---")

table_rows = []
percentage_rows = []

for idx, gas_name in enumerate(GASES):
    c_q1, c_q2, c_q3 = cnn_pred[idx*3 : idx*3 + 3]
    t_q1, t_q2, t_q3 = true_ret[idx*3 : idx*3 + 3]
    
    err_cnn = abs(c_q2 - t_q2)
    
    table_rows.append({
        "Target Molecule": gas_name,
        "Ground Truth (q2)": f"{t_q2:.3f}",
        "CNN Prediction (q2)": f"{c_q2:.3f} (Δ {err_cnn:.2f})",
        "Confidence Interval": f"[{c_q1:.2f}, {c_q3:.2f}]"
    })
    
    true_pct = (10 ** t_q2) * 100
    cnn_pct = (10 ** c_q2) * 100
    
    percentage_rows.append({
        "Target Molecule": gas_name,
        "True Composition": f"{true_pct:.5g}%",
        "Predicted Composition": f"{cnn_pct:.5g}%"
    })

summary_df = pd.DataFrame(table_rows).set_index("Target Molecule")
st.subheader("Chemical Abundance Matrix (log10 Abundance)")
st.dataframe(summary_df, use_container_width=True)

st.markdown("---")

pct_df = pd.DataFrame(percentage_rows).set_index("Target Molecule")
st.subheader("Actual Atmospheric Composition (%)")
st.markdown("*Translated from log abundances. Exoplanet atmospheres are heavily dominated by $H_2$ and $He$, so trace gas percentages are naturally microscopic.*")
st.dataframe(pct_df, use_container_width=True)