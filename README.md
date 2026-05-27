# ComboFNO: Fourier Neural Operator for Drug Combination Response Prediction

## Overview

ComboFNO is a deep learning framework for predicting the growth response of cancer cell lines to drug combinations, implemented in Python. It leverages Fourier Neural Operator (FNO) to capture spatial correlations in dose-response data, enabling accurate predictions across multiple experimental scenarios.The repository contains scripts for data preprocessing, Fourier neural network model, supporting two core prediction scenarios tailored to drug development needs.You can switch between different scenarios and predicted values in the main.py script.
1. new_drug: Predicting responses for combinations involving novel drugs.
2. new_combo: Predicting responses for novel drug pairs.

## Instructions

*data_processing.py*: Core data preprocessing script, including functions for dose-response matrix cleaning, logarithmic concentration transformation, combination data splitting, invalid sample filtering, and division of training set, validation set, and testing set.

*datasets.py*: Defines the custom PyTorch Dataset class (DrugResponseDataset) to load processed data, organize drug features, concentration coordinates and response values into model-input batches.

*fno_model.py*: Implements the Fourier Neural Operator (FNO) core model, including the FNO backbone, feature fusion layer and prediction head, supporting flexible adjustment of model width and Fourier modes.

*general_utils.py*: Provides utility functions such as random seed fixing, file name cleaning.

*train_eval.py*: Encapsulates the complete training and evaluation workflow, including model training loops, validation logic, early stopping mechanisms and best model saving.

*main.py*: Project entry script, responsible for loading configuration parameters, calling data processing and model training modules, and orchestrating the end-to-end execution of the entire prediction task.

*data*: Contains data_matrix_PERCENTGROWTHNOTZ and NCI-ALMANAC_drug_molformer_embeddings.
1. data_matrix_PERCENTGROWTHNOTZ/data_matrix_PERCENTGROWTH: Stored in Pickle format, this dataset is structured as a nested dictionary with a clear hierarchical organization. The outer dictionary uses cancer cell line names (e.g., "A549", "HCT116") as keys. Each value of the outer dictionary is another inner dictionary, where the keys are drug combination names (e.g., "Gemcitabine_Paclitaxel") and the values are 2D data matrices (shape: [N, 3], N represents the number of concentration pairs). The three columns of the matrix correspond to: Column 1 = Drug 1 concentration, Column 2 = Drug 2 concentration, Column 3 = Cancer cell growth response value (normalized to percentage). This structure integrates cell lines, drug combinations, and dose-response relationships.
2. NCI-ALMANAC_drug_molformer_embeddings: Stored in PyTorch Tensor format, this dataset provides molecular structure features of drugs from the NCI-ALMANAC database.

*The search range for parameters*: fno_modes:[4,8],fno_width:[12,24,36],fno_n_layers:[4,8],grid:[8,16],decoder_hidden_dims:[32,16],[64,32],[128,64],lifting:[32,64,128,256]. The best parameters for searching are in the CONFIG variable of the main.py script.

*result*: Stored in two scenarios, PERCENTGROWTHNOTZ and PERCENTGROWTH as pre trained comboFNO models for prediction values, as well as evaluation metrics for models in various cell lines.

## Dependencies

The code is developed with python 3.9.23
- numpy==2.0.2
- pandas==2.3.1
- scikit-learn==1.6.1
- torch==2.5.1
- scipy==1.13.1

comboFNO relies on the NeuralOP (neuraloperator) library for implementing the Fourier Neural Operator core. The official repository of NeuralOP is available at: https://github.com/neuraloperator/neuraloperator. This project is using neuraloperator version 1.0.2