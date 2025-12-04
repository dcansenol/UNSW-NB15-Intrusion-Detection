**UNSW-NB15 Intrusion Detection — Classical ML, Deep Learning, and Graph Embedding Approaches**

This repository contains the full implementation of a master's thesis project focusing on **intrusion and anomaly detection** using the **UNSW-NB15** dataset.
The study includes:

* **Classical machine learning models** (Logistic Regression, Random Forest, XGBoost)
* **Deep learning models** (MLP classifier, Autoencoder–based anomaly detection)
* **Graph-based feature learning** using **Node2Vec** embeddings

All experiments were conducted in Python using Jupyter Notebook. Both `.ipynb` source files and exported `.html` reports for each module are provided.

---

## Repository Structure

classic_ml/
    LR_RF_XGBoost.ipynb
    LR_RF_XGBoost.html

deep_learning/
    MLP_Autoencoder.ipynb
    MLP_Autoencoder.html

graph_embedding/
    graph_embedding.ipynb
    graph_embedding.html

---

## Installation

Clone the repository:

```bash
git clone https://github.com/dcansenol/UNSW-NB15-Intrusion-Detection.git
cd UNSW-NB15-Intrusion-Detection

## Install dependencies:

pip install -r requirements.txt

## Running the Notebooks

jupyter notebook

---

## **1. Classical Machine Learning (classic_ml/)**

Models implemented:

* Logistic Regression
* Random Forest
* XGBoost

Key features:

* Full preprocessing pipeline (imputation, scaling, encoding)
* Feature engineering (byte ratio, flow statistics, etc.)
* Model evaluation (accuracy, precision, recall, F1, ROC–AUC)
* SHAP-based feature importance for interpretability

---

## **2. Deep Learning (deep_learning/)**

Includes:

* **MLP classifier** (supervised)
* **Autoencoder anomaly detection** (unsupervised)

Main components:

* Layer-normalized MLP architecture
* Ablation studies (no engineered features, no byte_ratio, no flow_stats)
* Autoencoder reconstruction error thresholds
* Model training curves and evaluation metrics

---

## **3. Graph Embedding Approach (graph_embedding/)**

The UNSW-NB15 data is transformed into a **graph structure**:

* Nodes = protocol and service labels
* Edges = directed proto → service transitions
* Edge attributes = flow-based statistics

Node embeddings computed using:

* **Node2Vec** (biased random walks)
* Embedding dimensionality: 64 or 128
* Embeddings merged back into tabular dataset for classification

---

## **Citation**

If you use this work, please cite:

**"Intrusion and anomaly detection using classical machine learning, deep learning, and graph embeddings on the UNSW-NB15 dataset."**

---

Author

Doğan Can Şenol
MEng Computer Science
GISMA University of Applied Sciences
