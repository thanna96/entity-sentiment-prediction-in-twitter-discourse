# Graph-Based Entity Sentiment Prediction in Twitter Discourse

## Overview
This project explores entity-level sentiment analysis in Twitter data using graph neural networks (GNNs). Unlike traditional tweet-level sentiment classifiers, our approach incorporates relationships among users, entities, and tweets to improve classification accuracy and enable label-efficient sentiment propagation.

## Motivation
- Tweet-level sentiment misses nuanced opinions about specific entities.
- Prior models overlook the graph structure of social interactions.
- Graph Neural Networks (GNNs) like GraphSAGE and Relational GCN can leverage this relational context for improved performance.

## Dataset
**Twitter Entity Sentiment Analysis** (Kaggle)  
- https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data
- Nodes: Tweets, Entities, Users  
- Edges:
  - Tweet authored by user  
  - Tweet mentions entity  
  - Tweets co-mention same entity  

## Features
- Tweet nodes: BERT embeddings  
- Entity nodes: Pre-trained entity embeddings  
- User nodes: Aggregated sentiment vectors from prior tweets  

## Tasks
1. Sentiment Classification  
   Predict the sentiment toward each entity mentioned in tweets.
2. Label-Efficient Sentiment Propagation  
   Use minimal supervision to propagate sentiment across the graph.

## Methods
| Model Type        | Description |
|------------------|-------------|
| GraphSAGE         | Learns node representations via neighborhood sampling [Hamilton2017] |
| Relational GCN    | Incorporates edge types (e.g., mentions, authorship) [Fan2021] |
| GCN Propagation   | Semi-supervised label propagation using GCN or Laplacian methods [Kipf2017, Metscov2022] |
| Baseline          | BERT embeddings + MLP classifier (no graph structure) |

## Evaluation
- F1 Score vs. Model Type
- F1 Score vs. Labeled Data Percentage
- t-SNE / UMAP plots of node embeddings (before and after training)
- Sentiment heatmaps over time (if timestamped data available)

## Key Findings
- GNNs outperform non-graph baselines on entity-level sentiment classification.
- Semi-supervised label propagation is effective with very few labeled examples.
- Visualizations show clear sentiment-based and interaction-based clustering.

## Future Work
- Incorporate temporal modeling to capture evolving sentiment trends.
- Explore detection of coordinated or manipulative behavior in social discourse.

## References
- Chatterjee et al., COLING 2022 [ELSA2022]  
- Fan et al., AAAI 2021 [Fan2021]  
- Hamilton et al., NeurIPS 2017 [Hamilton2017]  
- Kipf & Welling, ICLR 2017 [Kipf2017]  
- Zhao et al., EMNLP 2022 [Zhao2022]  
- Cohen et al., NeurIPS D&B 2022 [Metscov2022]

## Project Structure


```text
.
├── data/            # Preprocessed graph data and raw input files
├── models/          # GNN model definitions (GraphSAGE, R-GCN, etc.)
├── notebooks/       # Jupyter notebooks for EDA and visualization
├── scripts/         # Training and evaluation scripts
├── utils/           # Graph construction, feature extraction, helpers
├── requirements.txt # Project dependencies
└── README.md        # Project overview
```

## Installation

1. Clone the repository:
git clone https://github.com/your-username/twitter-entity-sentiment-gnn.git
cd twitter-entity-sentiment-gnn

2. Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

## Usage

ToDo
