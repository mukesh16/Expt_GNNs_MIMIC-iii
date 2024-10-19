import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def authenticate_and_create_client(project_id):
    from google.colab import auth
    auth.authenticate_user()
    client = bigquery.Client(project=project_id)
    return client

def fetch_data(client, subject_ids):
    query = f"""
    SELECT
        p.subject_id,
        p.gender,
        p.dob,
        p.dod,
        p.expire_flag,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.admission_type,
        a.diagnosis,
        a.hospital_expire_flag,
        i.icustay_id,
        i.first_careunit,
        i.last_careunit,
        i.intime as icu_intime,
        i.outtime as icu_outtime,
        i.los as icu_los,
        d_icd.icd9_code as diagnosis_icd9_code,
        dicd.long_title as diagnosis_long_title,
        l.itemid as lab_itemid,
        l.charttime as lab_charttime,
        l.value as lab_value,
        l.valuenum as lab_valuenum,
        l.valueuom as lab_valueuom,
        dl.label as lab_item_label
    FROM
        `physionet-data.mimiciii_clinical.patients` p
        LEFT JOIN `physionet-data.mimiciii_clinical.admissions` a ON p.subject_id = a.subject_id
        LEFT JOIN `physionet-data.mimiciii_clinical.icustays` i ON a.hadm_id = i.hadm_id
        LEFT JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d_icd ON a.hadm_id = d_icd.hadm_id
        LEFT JOIN `physionet-data.mimiciii_clinical.d_icd_diagnoses` dicd ON d_icd.icd9_code = dicd.icd9_code
        LEFT JOIN `physionet-data.mimiciii_clinical.labevents` l ON a.hadm_id = l.hadm_id
        LEFT JOIN `physionet-data.mimiciii_clinical.d_labitems` dl ON l.itemid = dl.itemid
    WHERE
        p.subject_id IN ({','.join(map(str, subject_ids))})
    ORDER BY
        p.subject_id, a.admittime, i.intime, l.charttime;
    """
    df = client.query(query).to_dataframe()
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)
    
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])

    le_admission_type = LabelEncoder()
    df['admission_type'] = le_admission_type.fit_transform(df['admission_type'])

    le_diagnosis = LabelEncoder()
    df['diagnosis'] = le_diagnosis.fit_transform(df['diagnosis'])

    le_diagnosis_icd9_code = LabelEncoder()
    df['diagnosis_icd9_code'] = le_diagnosis_icd9_code.fit_transform(df['diagnosis_icd9_code'])

    le_lab_itemid = LabelEncoder()
    df['lab_itemid'] = le_lab_itemid.fit_transform(df['lab_itemid'])

    return df

def construct_graph(df):
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(f"patient_{row['subject_id']}", type='patient')
        G.add_node(f"admission_{row['hadm_id']}", type='admission')
        G.add_node(f"icu_{row['icustay_id']}", type='icu')

        G.add_node(f"diagnosis_{row['diagnosis_icd9_code']}", type='diagnosis')
        G.add_edge(f"admission_{row['hadm_id']}", f"diagnosis_{row['diagnosis_icd9_code']}", type='has_diagnosis')

        G.add_node(f"lab_{row['lab_itemid']}", type='lab')
        G.add_edge(f"admission_{row['hadm_id']}", f"lab_{row['lab_itemid']}", type='has_lab')

        G.add_edge(f"patient_{row['subject_id']}", f"admission_{row['hadm_id']}", type='has_admission')
        G.add_edge(f"admission_{row['hadm_id']}", f"icu_{row['icustay_id']}", type='has_icu')
    return G

def plot_graph(G):
    node_colors = {
        'patient': 'blue',
        'admission': 'green',
        'icu': 'red',
        'diagnosis': 'purple',
        'lab': 'orange'
    }
    edge_colors = {
        'has_admission': 'black',
        'has_icu': 'grey',
        'has_diagnosis': 'brown',
        'has_lab': 'pink'
    }

    node_color_list = [node_colors[G.nodes[node]['type']] for node in G.nodes]
    edge_color_list = [edge_colors[G.edges[edge]['type']] for edge in G.edges]

    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color_list, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title('Graph Representation of Medical Data')
    plt.show()

def encode_features(df):
    features = df.drop(columns=['subject_id', 'hadm_id', 'icustay_id', 'diagnosis'])
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(0)
    features = features.astype(float)
    return torch.tensor(features.values, dtype=torch.float)

def prepare_data_for_gcn(G, df):
    A = nx.to_scipy_sparse_array(G)
    X = encode_features(df)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    data = Data(x=X, edge_index=edge_index)
    data.y = torch.tensor(df['expire_flag'].values, dtype=torch.long)
    return data

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_model(model, data, train_mask, val_mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            val_loss = loss_fn(out[val_mask], data.y[val_mask])
            print(f'Epoch {epoch}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

def evaluate_model(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = out[test_mask].argmax(dim=1)
        accuracy = (preds == data.y[test_mask]).float().mean().item()
    return accuracy, preds, data.y[test_mask]

def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
