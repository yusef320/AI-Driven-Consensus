# AI-Driven-Consensus

## Overview
This repository contains Python scripts designed for generating synthetic datasets based on directed graphs and for training predictive models using these datasets. Two main scripts (`data_generation.py` and `model_training.py`) are provided for simulating graph dynamics and performing model comparisons.

## Scripts

### 1. data_generation.py
Generates synthetic datasets representing node state evolutions in directed graphs.

- **Graph Types Supported**:
  - Barabási–Albert preferential attachment (barabasi)
  - Watts–Strogatz small-world (watt)
  - Erdős–Rényi random graph (erdos)
- **Methods Implemented**:
  - Base method: Standard graph Laplacian-based dynamics.
  - Exponential method: Incorporates exponential decay into the adjacency matrix powers.
- **Generated Data**:
  - Node state evolutions (phi_values)
  - Final state mean values
  - Directed and bidirectional edge counts
- **Output**:
  - Training and testing datasets saved as .pt files.

**Usage**:
```
python data_generation.py
```

Datasets are stored in:
```
./data_directed/[case_type]_[graph_type]_n[n_nodes]_seq[seq_len]/train[train_samples]_test[test_samples]
```

### 2. model_training.py
Implements, trains, and evaluates multiple predictive models on the generated datasets:

- **Models Implemented**:
  - XGBoost
  - LSTM
  - Extended-LSTM (custom implementation)
  - Transformer
  - Convolutional LSTM
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - Average prediction time per sample
- **Features**:
  - Hyperparameter tuning for each model
  - Automatic GPU utilization (if available)
  - Results saved in a CSV file (results_all_models.csv)

**Usage**:
```
python model_training.py
```

## Requirements
- Python 3.8+
- PyTorch
- XGBoost
- NumPy
- NetworkX
- Scikit-learn

Install dependencies using:
```
pip install torch xgboost numpy networkx scikit-learn
```

## Folder Structure
```
.
├── data_generation.py
├── model_training.py
├── data_directed
│   ├── base_barabasi_n25_seq10
│   │   └── train800_test200
│   │       ├── train_data.pt
│   │       ├── train_targets.pt
│   │       ├── test_data.pt
│   │       └── test_targets.pt
│   └── ...
└── results_all_models.csv
```

## Acknowledgments
This work leverages GPU acceleration to efficiently simulate and analyze graph-based dynamics. Contributions to further enhance performance or add new models are welcome.

## License
This repository is open-source, licensed under the MIT License.
