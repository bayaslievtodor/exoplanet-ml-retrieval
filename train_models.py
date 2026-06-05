# IMPORTANT: FOR THIS TO WORK, IT NEEDS TO BE IN THE SAME DIRECTORY AS THE `FullDataset` FOLDER CONTAINING THE RAW DATA FROM
# THE ARIEL-2023 CHALLENGE. IF YOU'RE UNSURE WHERE TO FIND IT, DOWNLOAD FROM HERE: https://www.ariel-datachallenge.space/adc2023/

import os
import numpy as np
import torch
import joblib

from src.data import (load_raw_data, prepare_data,
                      split_and_scale_rf, split_scale_tensors_cnn,
                      CHEM_TARGETS, AUX_COLUMNS)
from src.models import ExoplanetCNN
from src.train import train_random_forest, train_cnn
from src.utils import set_seeds

# setings
DATA_DIR = "FullDataset/TrainingData"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "predictor_model_cnn.pt")
SPEC_SCALER_PATH = os.path.join(MODEL_DIR, "spec_scaler.pkl")
AUX_SCALER_PATH  = os.path.join(MODEL_DIR, "aux_scaler.pkl")

RANDOM_STATE = 123
TEST_SIZE = 0.2
BATCH_SIZE = 64
CNN_EPOCHS = 100
CNN_LR = 1e-3

# setup 
set_seeds(RANDOM_STATE)
os.makedirs(MODEL_DIR, exist_ok=True)

# step 0. Check if the data is even there. If not, terminate and link.

expected_path = os.path.join(DATA_DIR, "SpectralData.hdf5")
if not os.path.isfile(expected_path):
    print("=" * 60)
    print("  Full ARIEL dataset not found.")
    print("  The training data is not included in this repository due to its size.")
    print("  Download it from:")
    print("  https://www.ariel-datachallenge.space/adc2023/")
    print("  Extract the 'TrainingData' folder into the 'FullDataset' directory.")
    print("=" * 60)
    exit(1)

# 1. Load data 

print("Loading data...")
spectral, aux, labels = load_raw_data(DATA_DIR)
X, y, labelled_ids, wavelengths = prepare_data(spectral, aux, labels)
print(f"X shape: {X.shape}  y shape: {y.shape}")

# 2. Random Forest
print("\n" + "="*50)
print("Random Forest Baseline")
print("="*50)
X_train_scaled, X_val_scaled, y_train_rf, y_val_rf, scaler_rf = \
    split_and_scale_rf(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

rf_model, y_pred_rf, r2_scores = train_random_forest(
    X_train_scaled, y_train_rf, X_val_scaled, y_val_rf,
    random_state=RANDOM_STATE
)

# 3. 1D CNN
print("\n" + "="*50)
print("1D CNN")
print("="*50)
(train_loader, val_loader,
 spec_train_t, spec_val_t, aux_train_t, aux_val_t,
 y_train_t, y_val_t,
 spec_scaler, aux_scaler) = split_scale_tensors_cnn(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, batch_size=BATCH_SIZE
)

cnn_model = ExoplanetCNN()
print(f"CNN parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
cnn_model, train_losses, val_losses, y_pred_cnn, y_val_cnn, r2_cnn = \
    train_cnn(cnn_model, train_loader, val_loader,
              epochs=CNN_EPOCHS, lr=CNN_LR,
              print_every=10)

print("\nCNN R² scores per target:")
for name, score in zip(CHEM_TARGETS, r2_cnn):
    print(f"  {name:<20} {score:.3f}")

# 4. Save artefacts
print("\nSaving model and scalers...")
torch.save(cnn_model.state_dict(), MODEL_PATH)
joblib.dump(spec_scaler, SPEC_SCALER_PATH)
joblib.dump(aux_scaler,  AUX_SCALER_PATH)
print("Saved.")

# 5. Comparison summary
print("\n" + "="*50)
print("RF vs CNN R² Comparison")
print("="*50)
print(f"{'Target':<20} {'RF R²':>8} {'CNN R²':>8} {'Δ':>8}")
print("-" * 46)
for name, rf_s, cnn_s in zip(CHEM_TARGETS, r2_scores, r2_cnn):
    delta = cnn_s - rf_s
    print(f"{name:<20} {rf_s:>8.3f} {cnn_s:>8.3f} {delta:>+7.3f}")

molecules = ['H₂O', 'CO₂', 'CO', 'CH₄', 'NH₃']
rf_means = [np.mean(r2_scores[i*3:(i+1)*3]) for i in range(5)]
cnn_means = [np.mean(r2_cnn[i*3:(i+1)*3]) for i in range(5)]
print("\nMean R² per molecule:")
for mol, rf_m, cnn_m in zip(molecules, rf_means, cnn_means):
    print(f"  {mol}: RF={rf_m:.3f}  CNN={cnn_m:.3f}")

print("\nExperiment finished successfully.")