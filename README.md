# Medical Data Analysis with Graph Neural Networks

This project involves analyzing medical data using Graph Neural Networks (GCNs) with the MIMIC-III dataset. The data is fetched from Google BigQuery, preprocessed, and then used to construct a graph. A GCN model is trained and evaluated on this graph data to predict patient outcomes.

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

## License

This project is licensed under the GNU Affero General Public License V3. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The MIMIC-III dataset.
- PyTorch and PyTorch Geometric libraries.
- Google BigQuery and Google Colab.

## Author

[Your Name](https://github.com/mukesh16)
