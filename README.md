# DeepCDR-Hybrid
Transformer-enhanced multi-modal deep learning framework for cancer drug response prediction using hybrid drug representations and cross-attention fusion.

Abstract

In personalized medicine, predicting drug response in cancer patients requires integrating molecular drug structures with complex multi-omics profiles. Traditional computational models struggle to capture cross-modal interactions between chemical compounds and genomic features.

DeepCDR-Hybrid introduces a transformer-enhanced multi-modal deep learning framework featuring:

1.Transformer-based multi-head self-attention for multi-omics integration

2.Hybrid drug representation (ChemBERTa embeddings + Morgan fingerprints)

3.Bidirectional cross-attention fusion mechanisms

4.Integrated Explainable AI (XAI) components

Evaluated on the Cancer Cell Line Encyclopedia (CCLE) dataset containing 23,367 cell line–drug pairs, the model achieves: RMSE: 0.4289; Pearson Correlation: 0.9869; R² Score: 0.9721
53.7% improvement over DeepCDR baseline

This establishes a new benchmark in computational precision oncology.

Project Objectives

This project aims to:
Integrate gene expression, DNA methylation, and mutation data using transformer architectures,Combine structural and semantic drug representationsLearn complex cross-modal interactions,Achieve state-of-the-art,IC50 prediction performance,Provide explainable predictions for biomedical interpretability

Dataset

Primary dataset:

Cancer Cell Line Encyclopedia (CCLE)
Supplementary references from Genomics of Drug Sensitivity in Cancer (GDSC)
Dataset Statistics
23,367 cell line–drug response pairs
961 cancer cell lines
223 anti-cancer compounds

Multi-omics features:

Gene expression (RNA-seq)
DNA methylation
Mutation profiles

Target: Log-transformed IC50

Training Configuration

Framework: TensorFlow 2.x (Keras API)
Optimizer: Adam (LR = 1e-4)
Batch size: 32
Mixed Precision (FP16)
Early stopping (patience = 20)
Learning rate reduction
Seed = 42 (Reproducibility)
Hardware: NVIDIA GPU with CUDA acceleration

Results:-

<img width="404" height="317" alt="image" src="https://github.com/user-attachments/assets/9bbb6258-11dc-4761-beba-bbfb34da38fe" />

<img width="647" height="405" alt="image" src="https://github.com/user-attachments/assets/f4e4ad8c-499d-4365-bf3d-8e79446b2a01" />

✔ 53.7% RMSE reduction vs DeepCDR
✔ Strong generalization (minimal train-test gap)

Key Innovations

Transformer-based cross-omics modeling
Hybrid drug semantic + structural embeddings
Bidirectional cross-attention fusion
Advanced regularization for robust generalization
State-of-the-art predictive performance
