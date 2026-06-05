import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# Constants & column definitions

CHEM_TARGETS = [
    # Water Vapor (H2O)
    'log_H2O_q1', 'log_H2O_q2', 'log_H2O_q3',
    # Carbon Dioxide (CO2)
    'log_CO2_q1', 'log_CO2_q2', 'log_CO2_q3',
    # Carbon Monoxide (CO)
    'log_CO_q1',  'log_CO_q2',  'log_CO_q3',
    # Methane (CH4)
    'log_CH4_q1', 'log_CH4_q2', 'log_CH4_q3',
    # Ammonia (NH3)
    'log_NH3_q1', 'log_NH3_q2', 'log_NH3_q3'
]

N_SPECTRAL = 52 # number of wavelength channels
N_AUX = 8 # number of aux features

# The auxiliary column names (ordered)
AUX_COLUMNS = [
    'star_distance', 'star_mass_kg', 'star_radius_m', 'star_temperature',
    'planet_mass_kg', 'planet_orbital_period', 'planet_distance',
    'planet_surface_gravity'
]


# Raw data loading

def load_raw_data(data_dir="FullDataset/TrainingData"):

    spectral = h5py.File(f"{data_dir}/SpectralData.hdf5", "r")
    aux = pd.read_csv(f"{data_dir}/AuxillaryTable.csv", index_col='planet_ID')
    labels = pd.read_csv(f"{data_dir}/Ground Truth Package/QuartilesTable.csv", index_col='planet_ID')
    fm = pd.read_csv(f"{data_dir}/Ground Truth Package/FM_Parameter_Table.csv", index_col='planet_ID')

    return spectral, aux, labels, fm



# Clean & merge into X, y

def prepare_data(spectral, aux, labels):

    # Return:
    #    X : (n_planets, 60) – 52 spectral channels + 8 aux
    #    y : (n_planets, 15) – target log abundances
    #    labelled_ids : Index of planets used
    #    wavelengths : (52,) array of micron wavelengths
    
    # Keep only labelled planets (since the challenge thing was meant to be semi-supervised, only a fraction of the planets are labelled)
    labels_clean = labels.dropna()
    labelled_ids = labels_clean.index

    # Extract spectra from HDF5
    spectra_list = []
    for pid in labelled_ids:
        spectrum = spectral[f'Planet_{pid}']['instrument_spectrum'][:]
        spectra_list.append(spectrum)
    spectra_array = np.array(spectra_list)

    # Auxiliary features
    aux_labelled = aux.loc[labelled_ids].values #Exposing it raw so it's ready to append and not in a dataframe

    # Combine into full feature matrix
    X = np.hstack([spectra_array, aux_labelled]) # Merged

    # Target matrix
    y = labels_clean[CHEM_TARGETS].values 

    # Wavelengths (all planets share the same grid)
    sample_pid = labelled_ids[0]
    wavelengths = spectral[f'Planet_{sample_pid}']['instrument_wlgrid'][:]

    return X, y, labelled_ids, wavelengths



# Random Forest split & scaling (single scaler on full 60 features)

def split_and_scale_rf(X, y, test_size=0.2, random_state=123):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # fitting first one 
    X_val_scaled   = scaler.transform(X_val) # only transforming the second one to not deform the data
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler



# CNN split & separate scaling + PyTorch tensors & DataLoaders

def split_scale_tensors_cnn(X, y, test_size=0.2, random_state=123, batch_size=64):

    # Splits X into spectra (first 52 cols) and aux (last 8),
    # scales them separately, converts to tensors, returns DataLoaders
    # and the two fitted scalers.
 
    X_spectra = X[:, :N_SPECTRAL]
    X_aux     = X[:, N_SPECTRAL:]

    X_spec_train, X_spec_val, X_aux_train, X_aux_val, y_train, y_val = \
        train_test_split(X_spectra, X_aux, y, test_size=test_size, random_state=random_state)

    # Separate scaling
    spec_scaler = StandardScaler()
    aux_scaler  = StandardScaler()

    X_spec_train_s = spec_scaler.fit_transform(X_spec_train)
    X_spec_val_s   = spec_scaler.transform(X_spec_val)
    X_aux_train_s  = aux_scaler.fit_transform(X_aux_train)
    X_aux_val_s    = aux_scaler.transform(X_aux_val)

    # To PyTorch tensors
    def _tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    spec_train_t = _tensor(X_spec_train_s).unsqueeze(1)   # (N, 1, 52)
    spec_val_t   = _tensor(X_spec_val_s).unsqueeze(1)
    aux_train_t  = _tensor(X_aux_train_s)
    aux_val_t    = _tensor(X_aux_val_s)
    y_train_t    = _tensor(y_train)
    y_val_t      = _tensor(y_val)

    # DataLoaders
    train_ds = TensorDataset(spec_train_t, aux_train_t, y_train_t)
    val_ds   = TensorDataset(spec_val_t,   aux_val_t,   y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader,
            spec_train_t, spec_val_t, aux_train_t, aux_val_t,
            y_train_t, y_val_t,
            spec_scaler, aux_scaler)