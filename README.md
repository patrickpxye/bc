# Bridged Clustering (BC)

This repo contains code for our paper  
**"Bridged Clustering for Representation Learning: Semi-Supervised Sparse Bridging".**

## Overview
Bridged Clustering (**BC**) learns predictors from *unpaired* input-only data (𝒳), *unpaired* output-only data (𝒴), and a small set of paired examples (𝒮).  
- Cluster 𝒳 and 𝒴 independently.  
- Learn a **sparse, interpretable cluster-to-cluster bridge** from a few paired examples.  
- Predict by mapping a new input to its nearest input cluster and returning the centroid of the linked output cluster.  

BC is **model-agnostic**, **label-efficient** (1–4 pairs per cluster), and scales **linearly** once cluster count is small relative to dataset size.  
It supports **bidirectional prediction** (X→Y and Y→X).

## Experiments
We evaluate on four multimodal datasets:
- **BIOSCAN-5M**: insect images ↔ DNA barcodes  
- **WIT**: Wikipedia images ↔ captions  
- **Flickr30k**: everyday images ↔ captions  
- **COCO**: object images ↔ captions  

Baselines include SSL methods (MeanTeacher, FixMatch, LapRLS, TSVR, GCN, UCVME, RankUp), unmatched regression (KMM, EM), and output-aware methods (EOT, GW).

## Running
1. Install dependencies (Python 3.10+, PyTorch, scikit-learn, POT, PyG).  
2. Prepare data splits (`input-only`, `output-only`, and small `supervised` set).  
3. Run experiments
