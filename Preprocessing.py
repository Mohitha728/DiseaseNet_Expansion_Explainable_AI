import pandas as pd
import os

# --- Set file paths ---
FOLDER = "C:/DiseaseNet_Project/Final"
TCGA_FILE = os.path.join(FOLDER, "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz")
ANNOT_FILE = os.path.join(FOLDER, "probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy")
DISEASE_FILES = [
    "GSE36064_series_matrix.txt",  # Asthma
    "GSE38291_series_matrix.txt",  # Diabetes
    "GSE42861_series_matrix.txt",  # Arthritis
    "GSE84727_series_matrix.txt",  # Schizophrenia
    "GSE88824_series_matrix.txt"   # Obesity
]
DISEASE_FILES = [os.path.join(FOLDER, fn) for fn in DISEASE_FILES]

# --- Load annotation file ---
annot = pd.read_csv(ANNOT_FILE, sep='\t', low_memory=False)
probe2gene = dict(zip(annot['#id'], annot['gene']))

# --- Helper function: aggregate probes to gene level ---
def probes_to_genes(df, probe2gene):
    probes = [p for p in df.index if p in probe2gene]
    df_sub = df.loc[probes]
    df_sub['gene'] = [probe2gene[p] for p in df_sub.index]
    df_sub = df_sub[df_sub['gene'].notnull() & (df_sub['gene'] != ".")]
    df_agg = df_sub.groupby('gene').mean()
    return df_agg

# --- TCGA processing (first 5000 probes, 1000 samples) ---
print("Loading TCGA methylation matrix (first 5000 probes and 1000 samples)...")
tcga = pd.read_csv(TCGA_FILE, sep='\t', low_memory=False, index_col=0, nrows=5000)
tcga = tcga.iloc[:, :1000]
tcga_genes = probes_to_genes(tcga, probe2gene)
print("TCGA gene-level shape (subset):", tcga_genes.shape)

# --- Disease dataset processing (first 10000 probes, 1000 samples) ---
disease_gene_dfs = []
for f in DISEASE_FILES:
    print(f"Loading {f} (first 10000 rows and 1000 columns)...")
    df = pd.read_csv(f, sep='\t', low_memory=False, index_col=0, comment='!', nrows=10000)
    df = df.iloc[:, :1000]
    df = df.dropna(axis=1, thresh=int(0.8 * df.shape[0]))
    gene_df = probes_to_genes(df, probe2gene)
    print(f"Gene-level shape for {f} (subset):", gene_df.shape)
    disease_gene_dfs.append(gene_df)

# --- Find shared genes across all valid sets (exclude empty disease sets) ---
shared_genes = set(tcga_genes.index)
for dg in disease_gene_dfs:
    if dg.shape[0] > 0 and dg.shape[1] > 0:
        shared_genes &= set(dg.index)
shared_genes = sorted(list(shared_genes))
print("Total shared genes:", len(shared_genes))

# --- Final output: write only samples × gene matrix ---
tcga_final = tcga_genes.loc[shared_genes].T  # samples × genes
tcga_final.to_csv(f"{FOLDER}/Processed_TCGA_for_CancerNet_final.csv")

for i, dg in enumerate(disease_gene_dfs):
    # Only write non-empty sets
    if dg.shape[0] > 0 and dg.shape[1] > 0:
        dg_final = dg.loc[shared_genes].T  # samples × genes
        print(f"Saving Disease {i+1}, shape:", dg_final.shape)
        dg_final.to_csv(f"{FOLDER}/Processed_Disease_{i+1}_final.csv")
    else:
        print(f"Disease {i+1} has no valid samples/features; skipping.")

print("Preprocessing complete. Outputs are guaranteed to have samples × gene feature matrices.")
