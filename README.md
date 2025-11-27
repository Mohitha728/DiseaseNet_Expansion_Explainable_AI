# DiseaseNet_Expansion_Explainable_AI

## Project Overview
Developed **DiseaseNet**, a transfer learning pipeline for multi-class classification of non-communicable diseases (asthma, rheumatoid arthritis, type 2 diabetes, obesity) using DNA methylation data as biomarkers.


### Data Pipeline
- **Sourced & Harmonized**: TCGA pan-cancer (pretraining) + 4 GEO NCD cohorts (GSE77702 asthma, GSE42861 arthritis, GSE48472 diabetes, GSE59065 obesity)
- **Preprocessing**: Probe→gene aggregation (Illumina annotation), missing value imputation, Z-score normalization, SMOTE class balancing
- **Feature Engineering**: Gene intersection across datasets, handled high-dimensionality (1000s CpG sites → shared genes)

### Model Development (2-Stage Deep Learning)
```
Stage 1: Autoencoder Pretraining on TCGA
- Encoder: Dense(256→128 ReLU) → 128D latent space
- Decoder: Reconstructs methylation profiles (MSE loss)

Stage 2: Transfer Learning Classifier
- Frozen encoder → Dropout(0.3) → Dense(64) → Softmax (4 classes)
- Fine-tuned entire model (categorical cross-entropy)
```
**Outperformed RandomForest benchmark** with near-perfect test accuracy/F1-scores

### Key Innovations
- **Transfer Learning**: Cancer pretraining → NCD adaptation (solves limited NCD samples)
- **Class Balancing**: SMOTE eliminated arthritis dominance bias
- **Validation**: 80/20 train-test split + 10% validation, model checkpointing

### Explainable AI (SHAP Analysis)
**Identified top biomarkers per disease**:
```
Diabetes: TSPAN33, TOR3A, RFC5, PAQR6
Asthma: SMAP1, APAF1, ARAP1, GJD4  
Arthritis: SMAP1, CCHCR1, HBEGF, STAT1
Obesity: TCF19, CCHCR1, HBEGF, IKBP
```
Generated confusion matrices, SHAP plots, classification reports
