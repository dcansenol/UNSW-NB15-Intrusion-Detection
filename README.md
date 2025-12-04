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
```

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

# **Glossary of Terms**

## **1. Dataset and Common Variables**

* **train_path / test_path** — Paths of the UNSW-NB15 CSV files.
* **train / test** — Raw DataFrames for training and testing.
* **df_train / df_test** — DataFrames used in the deep learning module.
* **df** — Temporary DataFrame used in helper functions.
* **PROJECT_ROOT** — Root directory of the repository.
* **RANDOM_SEED** — Fixed seed (42) for reproducibility.
* **LEAKAGE_COLS** — Columns removed to prevent data leakage (id, IPs, ports, etc.).
* **TARGET_COL** — Name of the binary label column (`label`).
* **ALT_TARGET** — Alternative label column (`attack_cat`).
* **FEATURES** — List of usable feature columns.
* **keep_text** — Columns kept as text categories.
* **X_train / X_test** — Feature matrices for classical ML models.
* **y_train / y_test** — Label vectors for classical ML models.
* **X_tr / X_te** — Preprocessed feature matrices for MLP/Autoencoder.
* **y_tr / y_te** — Labels for deep learning models.
* **X_tr_raw / X_te_raw** — Features after removing leakage columns but before preprocessing.
* **X_tr_ge / X_te_ge** — Feature matrices built from graph embeddings.
* **y_tr_ge / y_te_ge** — Labels for graph-based models.

---

## **2. Label Functions**

* **build_label(df)** — Converts `attack_cat` into a binary label (Normal → 0, Attack → 1).
* **to_label(df)** — Same logic, used in the graph embedding module.

---

## **3. Graph and Node2Vec Terms**

* **SRC / DST** — Columns representing graph nodes (`proto` and `service`).
* **train_edges / test_edges** — Source–destination edge lists.
* **avail / need / missing** — Sets used to check required columns.
* **G** — Directed graph created from the training edges.
* **n2v_model** — Trained Node2Vec embedding model.
* **dim** — Size of the embedding vector.
* **get_vec(node)** — Returns the embedding of a graph node.
* **make_edge_features(df_src_dst)** — Builds feature vectors by combining src/dst embeddings.
* **dimensions / walk_length / num_walks / workers / p / q / seed** — Node2Vec parameters.

---

## **4. Preprocessing Components**

* **numeric** — Imputation + scaling steps for numeric columns.
* **categor** — Imputation + one-hot encoding for categorical columns.
* **num_cols / cat_cols** — Lists of numerical and categorical column names.
* **pre / pre_local / preprocessor** — Combined preprocessing transformer.
* **drop_cols** — Columns removed before preprocessing.
* **make_preprocessor()** — Creates the full preprocessing pipeline.

---

## **5. MLP Classifier Terms**

* **build_mlp(input_dim)** — Creates the MLP network.
* **mlp** — The MLP classifier model.
* **history** — Training history produced by `.fit()`.
* **y_pred_proba** — Probability predictions.
* **y_pred** — Binary predictions (0/1).
* **Dense / BatchNormalization / Dropout** — Layers used in the model.

---

## **6. Autoencoder Terms**

* **build_autoencoder(input_dim)** — Creates the encoder–decoder architecture.
* **ae** — Trained autoencoder model.
* **X_tr_normal** — Samples with label 0 used to train the autoencoder.
* **history_ae** — Autoencoder training history.
* **reconstructed** — Autoencoder outputs for test samples.
* **reconstruction_error** — MSE between input and reconstruction.
* **fpr / tpr / roc_auc_ae** — ROC curve and AUC terms.

---

## **7. Classical ML Models**

* **numeric_pipe** — Numeric preprocessing steps.
* **categorical_pipe** — Categorical preprocessing steps.
* **preprocess** — Combined transformer for all models.
* **make_pipeline(estimator)** — Wraps a model with preprocessing.
* **LogisticRegression** — Linear baseline classifier.
* **RandomForestClassifier** — Tree-based ensemble model.
* **XGBClassifier** — Gradient-boosting model from XGBoost.
* **models** — Dictionary of all ML model pipelines.
* **fitted_models** — Dictionary of trained models.
* **scale_pos_weight** — Balances class weights for XGBoost.

---

## **8. Feature Engineering & Ablation Terms**

* **X_train_fe / X_test_fe** — Datasets after applying feature engineering.
* **byte_ratio** — Feature computed as `(sbytes + 1) / (dbytes + 1)`.
* **make_fe_pipeline(...)** — Preprocessing pipeline that supports engineered features.
* **train_rf(...)** — Trains Random Forest on FE-modified data.
* **X_train_base / X_test_base** — Baseline datasets without FE.
* **X_train_no_ratio / X_test_no_ratio** — Version without the `byte_ratio` feature.
* **X_train_no_flow / X_test_no_flow** — Version without flow-related features.
* **num_cols_* / cat_cols_*** — Column lists for each dataset version.
* **scenarios** — All dataset variations used in ablation.
* **results_ablation** — Stores performance metrics for each scenario.
* **ablation_df** — Summary table for ablation results.

---

## **9. Evaluation Terms**

* **evaluate_model(...)** — Computes accuracy, precision, recall, f1, and ROC-AUC.
* **accuracy_score** — Percentage of correct predictions.
* **precision_score** — How many predicted attacks were correct.
* **recall_score** — How many real attacks were detected.
* **f1_score** — Harmonic mean of precision and recall.
* **roc_auc_score** — Measures class separation performance.
* **classification_report** — Detailed table of metrics.
* **confusion_matrix** — TP, FP, TN, FN counts.
* **fpr / tpr** — Values used for ROC curve plots.
* **RocCurveDisplay** — Utility for visualizing ROC curves.

---

## **10. Graph Embedding + Logistic Regression Terms**

* **clf** — Logistic Regression model trained on graph-based embeddings.
* **y_pred** — Predicted labels from the graph model.
* **y_prob** — Predicted probabilities.
* **auc** — ROC-AUC score for the graph-based classifier.
* **"GraphEmb+LR"** — Name used in ROC visualizations.

---

## **11. Plotting Terms**

* **plt** — `matplotlib.pyplot`.
* **plot_hist(...)** — Helper for drawing histograms.
* **bins** — Number of histogram bins.
* **alpha** — Transparency level in plots.
* **label** — Legend name for plots.

---

## **Citation**

If you use this work, please cite:

**"Intrusion and anomaly detection using classical machine learning, deep learning, and graph embeddings on the UNSW-NB15 dataset."**

---

## Author

**Doğan Can Şenol**  
MEng Computer Science  
GISMA University of Applied Sciences

