# Harnessing EHR Data: A Graph-Based Model for Predicting Patient Criticalness with Graph Neural Networks

## Project Overview
  This project aims to utilize Electronic Health Records (EHR) and Graph Neural Networks (GNNs) to predict patient criticalness, with a specific focus on mortality prediction. By representing patient data as graphs, this approach uncovers the complex relationships between clinical, demographic, and ICU data, improving prediction accuracy over traditional machine learning models. The project integrates patient similarity models with GNNs to forecast critical outcomes such as patient deterioration or death.

## Problem Statement
  Early detection of patient criticalness is critical for improving healthcare outcomes. With increasing ICU admissions and complex medical histories, predicting patient mortality and critical conditions is essential. Current traditional models fail to fully utilize the interdependencies between different data points in Electronic Health Records (EHR), leading to suboptimal predictions. This project aims to fill this gap by using Graph Neural Networks to model and predict patient outcomes based on integrated EHR data.

## Objectives
- Prediction of Patient Mortality and Criticality: Develop a model to predict whether a patient is at risk of critical outcomes using GNNs.
- Graph Representation of Patient Data: Transform patient data from multiple sources (e.g., diagnoses, lab results, prescriptions) into graph structures for enhanced prediction capabilities.
- Improved Accuracy: Leverage GNNs to better understand complex relationships between clinical variables, improving model accuracy in predicting mortality.

## Key Features
1. Graph-Based Approach: Converts EHR data into graph structures, with nodes representing patients and edges representing relationships based on clinical similarities.
2. Patient Similarity Modeling: Establishes patient similarity based on clinical history, lab results, diagnoses, prescriptions, and ICU stays.
3. Graph Neural Network (GNN): Uses PyTorch Geometric and Graph Convolutional Networks (GCNs) to train and predict patient outcomes from graph-structured data.
4. Binary Classification for Mortality Prediction: Predicts mortality risk based on a binary classification approach, with binary cross-entropy loss as the objective function.

## Data Sources
- MIMIC-III Database: A critical care database containing de-identified health data for over 40,000 patients, including clinical variables like vital signs, lab results, prescriptions, and diagnoses.
- PhysioNet: A collection of freely available data, including clinical and ICU data from various sources (for additional model training and validation).


## Research Methodology
```
1. Data Preprocessing
  - Data Cleaning: Handle missing values, outliers, and normalize features.
  - Feature Engineering: Extract key features from various EHR tables such as:
  - Patient Demographics: Age, gender, ethnicity.
  - Clinical Data: Lab results, medical history (binary vectors for diagnoses).
  - ICU Data: Length of stay, treatment procedures.
  - Prescription Data: Medications and dosages.

2. Graph Construction
  - Nodes: Each patient is represented as a node, with features that describe the patient’s clinical, demographic, and treatment data.
  - Edges: Relationships between patients are defined based on shared diagnoses, co-occurrence in ICU stays, or similarity in treatment.
  - Graph Representation: The data is represented as a heterogeneous graph, where patient features and relationships are captured to facilitate predictive analysis.

3. Model Architecture
  - Graph Neural Networks (GNNs): The model architecture uses Graph Convolutional Networks (GCNs), a specific type of GNN, to process the graph structure.
  - Model Training: The network learns from the graph structure and predicts the likelihood of mortality using binary classification.
  - Loss Function: The model uses binary cross-entropy loss to train the network, optimizing the prediction of patient mortality.

4. Evaluation
  - Metrics: The model is evaluated using standard performance metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness in predicting patient mortality.
  - Comparison with Traditional Models: The performance of the GNN-based model is compared with traditional machine learning methods such as logistic regression and random forests to validate the improvements brought by the graph-based approach.
```

## Project Structure

```
├── lib
│   └── utils.py
├── notebook
│   └── BigQuery_df_fuc.ipynb
├── README.md
```

## Requirements

1. Google Cloud account with BigQuery API access.
2. Google Colab account to run the Jupyter notebook.
3. Google Drive to store and access the utility script.

## Setup

### Step 1: Clone the Repository

Clone this repository to your local machine:

```sh
git clone https://github.com/mukesh16/Expt_GNNs_MIMIC-iii.git
cd Expt_GNNs_MIMIC-iii
```

### Step 2: Install Dependencies

Make sure you have the necessary Python packages installed. You can install them using the following commands:

```sh
!pip install --upgrade google-cloud-bigquery
!pip install torch_geometric
!pip install google-cloud-bigquery-storage==2.18.1
!pip install --upgrade google-cloud-bigquery[pandas]
```

### Step 3: Upload `lib` Folder to Google Drive

Upload the `lib` folder containing `utils.py` to your Google Drive.

### Step 4: Open and Run the Colab Notebook

1. Open the Colab notebook: `notebook/BigQuery_df_fuc.ipynb`.
2. Mount your Google Drive within the notebook.
3. Replace the path to the `lib` folder in the notebook with the actual path in your Google Drive.
4. Set the parameters (such as `subject_ids`, `epochs`, and `learning_rate`) in the notebook.
5. Run the cells in the notebook.

## Project Details

[![Patient Similarity Model Flowchart](sandbox:/mnt/data/graph_neural_network.jpg)](https://lucid.app/documents/embedded/4f20d1f6-9123-4e9d-8265-d1e509370987)


### `lib/utils.py`

This script contains the following functions:

- `authenticate_and_create_client(project_id)`: Authenticates and creates a BigQuery client.
- `fetch_data(client, subject_ids)`: Fetches data from BigQuery based on provided subject IDs.
- `preprocess_data(df)`: Preprocesses the data, including filling missing values and encoding categorical variables.
- `construct_graph(df)`: Constructs a graph from the preprocessed data.
- `plot_graph(G)`: Plots the constructed graph.
- `encode_features(df)`: Encodes features for the GCN model.
- `prepare_data_for_gcn(G, df)`: Prepares data for the GCN model.
- `GCN(in_channels, out_channels)`: Defines the GCN model.
- `train_model(model, data, train_mask, val_mask, epochs, lr)`: Trains the GCN model.
- `evaluate_model(model, data, test_mask)`: Evaluates the GCN model.
- `plot_confusion_matrix(true_labels, predicted_labels)`: Plots the confusion matrix for model predictions.

### `notebook/BigQuery_df_fuc.ipynb`

This notebook guides you through:

- Authenticating and creating a BigQuery client.
- Fetching and preprocessing data.
- Constructing and plotting a graph.
- Preparing data for the GCN model.
- Defining, training, and evaluating the GCN model.
- Plotting the confusion matrix to visualize model performance.

## Acknowledgements

- MIMIC-III Database: The dataset is provided by PhysioNet, a resource for large, openly accessible clinical datasets.
- PyTorch Geometric: The graph neural network library used for building and training the GNN model.
- Google BigQuery and Google Colab.

## References

1. GraphEHR: Heterogeneous Graph Neural Network for Electronic Health Records - [Liu, Z., Li, X., Peng, H., He, L., & Yu, P. S. (2021). Heterogeneous Similarity Graph Neural Network on Electronic Health Records. ArXiv.](https://arxiv.org/abs/2101.06800).
2. MIMIC-III Database - [Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet.](https://doi.org/10.13026/C2XW26.)
3. Graph Neural Networks: A Survey by Zhou et al. (2018) - [Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2018). Graph Neural Networks: A Review of Methods and Applications. ArXiv. ](https://arxiv.org/abs/1812.08434).
Lucidchart Model Link: Graph-Based Patient Similarity Model

## License

This project is licensed under the GNU Affero General Public License V3. See the [LICENSE](LICENSE) file for details.

## Author

[Mukesh Kumar Sahu](https://github.com/mukesh16)
