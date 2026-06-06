import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

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
    return cnn, joblib.load(SPEC_SCALER_PATH), joblib.load(AUX_SCALER_PATH)

cnn_model, spec_scaler, aux_scaler = load_assets()
df = pd.read_csv(CSV_PATH, index_col='planet_ID')

planet_mapping = dict(zip(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], df.index.tolist()[:5]))

st.title("Exoplanet Evaluation Demo")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    selected_nickname = st.selectbox("Select Target Planet", list(planet_mapping.keys()))
with col2:
    show_rationale = st.toggle("Visualize Model Rationale")

row = df.loc[planet_mapping[selected_nickname]]

spec_raw = row[SPECTRAL_COLS].values.astype(float).reshape(1, -1)
aux_raw = row[AUX_COLS].values.astype(float).reshape(1, -1)
true_ret = row[TARGET_COLS].values.astype(float)

spec_t = torch.tensor(spec_scaler.transform(spec_raw), dtype=torch.float32).unsqueeze(1)
spec_t.requires_grad_(True)
aux_t = torch.tensor(aux_scaler.transform(aux_raw), dtype=torch.float32)

cnn_pred = cnn_model(spec_t, aux_t)

meta_cols = st.columns(4)
meta_cols[0].metric("Surface Gravity", f"{row['planet_surface_gravity']:.2f} m/s²")
meta_cols[1].metric("Orbital Period", f"{row['planet_orbital_period']:.2f} Days")
meta_cols[2].metric("Stellar Temperature", f"{int(row['star_temperature'])} K")
meta_cols[3].metric("Distance", f"{row['planet_distance']:.3f} AU")

st.markdown("---")
st.subheader("52-Channel Transit Depth Spectrum")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(range(1, 53), spec_raw[0], color="#1f77b4", linewidth=2, label="Transit Depth")

if show_rationale:
    gas_idx = st.selectbox("Select Gas to Explain", range(len(GASES)), format_func=lambda x: GASES[x])
    target_output = cnn_pred[0, gas_idx*3 + 1]
    target_output.backward()
    saliency = spec_t.grad.abs().squeeze().detach().numpy()
    
    norm_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-9)
    ax.scatter(range(1, 53), spec_raw[0], c=norm_saliency, cmap='Reds', s=50, zorder=5, label="Importance")
    ax.set_title(f"Model Attention for {GASES[gas_idx]}")

ax.set_xlim(1, 52)
ax.set_xlabel("Channel")
ax.set_ylabel("Transit Depth")
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

st.markdown("---")

table_rows, percentage_rows = [], []
cnn_pred_flat = cnn_pred.detach().numpy().flatten()

for idx, gas_name in enumerate(GASES):
    c_q1, c_q2, c_q3 = cnn_pred_flat[idx*3 : idx*3 + 3]
    t_q2 = true_ret[idx*3 + 1]
    
    table_rows.append({
        "Target Molecule": gas_name,
        "Ground Truth (q2)": t_q2,
        "CNN Prediction (q2)": c_q2,
        "Δ (Error)": abs(c_q2 - t_q2),
        "Confidence Interval": f"[{c_q1:.2f}, {c_q3:.2f}]",
        "In Bounds?": "Y" if c_q1 <= t_q2 <= c_q3 else "N"
    })
    
    percentage_rows.append({
        "Target Molecule": gas_name,
        "True Composition (%)": (10 ** t_q2) * 100,
        "Predicted Composition (%)": (10 ** c_q2) * 100
    })

st.subheader("Chemical Abundance Matrix (log10 Abundance)")
st.dataframe(
    pd.DataFrame(table_rows).set_index("Target Molecule").style.background_gradient(subset=['Δ (Error)'], cmap='YlOrRd'),
    use_container_width=True,
    column_config={
        "Ground Truth (q2)": st.column_config.NumberColumn(format="%.3f"),
        "CNN Prediction (q2)": st.column_config.NumberColumn(format="%.3f"),
        "Δ (Error)": st.column_config.NumberColumn(format="%.2f"),
        "In Bounds?": st.column_config.TextColumn()
    }
)

st.markdown("---")
st.subheader("Actual Atmospheric Composition (%)")
st.dataframe(
    pd.DataFrame(percentage_rows).set_index("Target Molecule"), 
    use_container_width=True,
    column_config={
        "True Composition (%)": st.column_config.NumberColumn(format="%.5g"),
        "Predicted Composition (%)": st.column_config.NumberColumn(format="%.5g")
    }
)