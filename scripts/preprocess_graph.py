## Parse raw tweet and entity data
#
# Build a heterogeneous graph using torch-geometric or networkx
#
# Encode node features:
#
# Tweet: use BERT (transformers) to generate embeddings
#
# Entity: use pre-trained embeddings or one-hot vectors
#
# User: aggregate sentiment from authored tweets
#
# Define and save graph objects to data/processed_graph.pt