# Exoplanet Atmospheric Retrieval via 1D CNN

A simple ML model for predicting a exoplanet's atmospheric composition based on spectroscopic and astronomical data. 
Uses the simulated data from the ARIEL-2023 Data Space Challenge: https://www.ariel-datachallenge.space/adc2023/

## Core Idea

Traditionally, mapping spectral data to atmospheric chemical abundances requires computationally heavy forward models, running millions of simulations per observation. As next-generation telescopes gather increasingly massive datasets, this method faces severe scaling bottlenecks.

This project demonstrates an ML prototype as an alternative - specifically 1D Convolutional Neural Network (CNN). By fusing spectral time-series data with planetary/stellar metadata, the model bypasses traditional simulations to predict log10 chemical abundances and their uncertainties in milliseconds.

## Design Document

The design document contains thorough information and documentation behind the architectural choices, including the design path from tabular to neural models and an in-depth performance evaluation. 

[**View Design Doc PDF**](./design_document.pdf)

## Live Mini-Demo

A simple, interactive demo of the 1D CNN model performing inference on validation data can be accessed here:
https://exoplanet-ml-retrieval-icjapra39vletquko9sgj3.streamlit.app/#chemical-abundance-matrix-log10-abundance

## Architecture & Design Path

* Baseline Model: Random Forest (evaluated and discarded due to the "neighbor-blind" nature of tabular models - spectral channels were read in isolation).
* Primary Model: 1D CNN + MLP Dual-Pathway Fusion.
  * Pathway 1: 1D Convolutions process 52-channel spectral data to capture spatial/frequency relationships (e.g., broad absorption signatures).
  * Pathway 2: A Multilayer Perceptron (MLP) encodes auxiliary data (star radius, planet mass, etc.).
* Loss Function Pinball Loss is used  to predict quantiles (q1, q2, q3), providing built-in uncertainty estimation in respect to the training data variables.
* **Target Molecules: H2O, CO2, CO, CH4, NH3.

## Repo Structure

* /src/ - Data loading,  model definition and training, util functions.
* /models/ - .pt model and scikit-learn scalers.
* /demo/ - Tiny Streamlit application to show the model in acton & five random planets from the testing data chosen from the ARIEL 2023 dataset.
* design_document.pdf - The full accompanying design document
* requirements.txt - Python lib dependencies
* train_models.py - The Python script that trains the models from scratch (but with a set seed 123 for reproducibility).

## Training The Models Yourself

Keep in mind - the repository alone is not enough to train the models, you will also need the raw data.
You may find the data here: https://www.ariel-datachallenge.space/adc2023/
**Important:** Simply extract the folder "FullDataset" in the main directory exactly the way it came packaged. Don't rename or take anything out.
