import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import shap  # Do NOT name this script shap.py!

# --- Data Loading & Harmonization ---
FOLDER = "C:/DiseaseNet_Project/Final"
TCGA_FILE = f"{FOLDER}/Processed_TCGA_for_CancerNet_final.csv"
disease_files = [
    "Processed_Disease_1_final.csv",  # Asthma
    "Processed_Disease_2_final.csv",  # Diabetes
    "Processed_Disease_3_final.csv",  # Arthritis
    "Processed_Disease_5_final.csv"   # Obesity
]
disease_names = ["Asthma", "Diabetes", "Arthritis", "Obesity"]

tcga = pd.read_csv(TCGA_FILE, index_col=0)
gene_sets = [set(pd.read_csv(FOLDER + "/" + f, index_col=0).columns) for f in disease_files]
shared_genes = set(tcga.columns)
for gs in gene_sets:
    shared_genes &= gs
shared_genes = list(shared_genes)
tcga = tcga[shared_genes]

# --- Combine Disease Cohorts ---
X, y = [], []
disease_counts = []
for f, name in zip(disease_files, disease_names):
    df = pd.read_csv(FOLDER + "/" + f, index_col=0)[shared_genes]
    if df.shape[0] > 0:
        X.append(df.values)
        y += [name] * df.shape[0]
        disease_counts.append(df.shape[0])
    else:
        disease_counts.append(0)
X = np.concatenate(X, axis=0)
le = LabelEncoder()
y_num = le.fit_transform(y)
y_cat = to_categorical(y_num)

# -- Class distribution visualization
plt.figure(figsize=(7,4))
plt.bar(disease_names, disease_counts)
plt.ylabel("Number of Samples")
plt.title("Class Balance Across Disease Cohorts")
plt.tight_layout()
plt.savefig(f"{FOLDER}/disease_class_balance.png", dpi=150)
plt.close()

# --- Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
tcga_imputed = imputer.fit_transform(tcga.values)

# --- Scaling
scaler = StandardScaler()
tcga_scaled = scaler.fit_transform(tcga_imputed)
X_scaled = scaler.transform(X_imputed)

# --- Rectify class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y_num)
y_balanced_cat = to_categorical(y_balanced)

unique, counts = np.unique(y_balanced, return_counts=True)
plt.figure(figsize=(7,4))
plt.bar(le.inverse_transform(unique), counts)
plt.ylabel("Number of Samples")
plt.title("Balanced Class Distribution via SMOTE")
plt.tight_layout()
plt.savefig(f"{FOLDER}/balanced_class_distribution.png", dpi=150)
plt.close()

# --- Split data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced_cat, test_size=0.2, random_state=42)

# --- Deep Learning: Autoencoder + classifier
input_dim = tcga_scaled.shape[1]
encoding_dim = 128
input_layer = Input(shape=(input_dim,))
encoder = Dense(256, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu', name="latent")(encoder)
decoder = Dense(256, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(tcga_scaled, tcga_scaled, epochs=10, batch_size=32, validation_split=0.1)

encoded_input = Input(shape=(input_dim,))
x = autoencoder.layers[1](encoded_input)
x = autoencoder.layers[2](x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(y_train.shape[1], activation='softmax')(x)
classifier = Model(inputs=encoded_input, outputs=output)

for layer in classifier.layers[:3]:
    layer.trainable = False
checkpoint = ModelCheckpoint(f"{FOLDER}/DiseaseNet_best_multiclass_model_balanced.h5", monitor='val_loss', save_best_only=True)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[checkpoint])
for layer in classifier.layers:
    layer.trainable = True
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

loss, acc = classifier.evaluate(X_test, y_test)
print(f"Best Deep Model Test Accuracy: {acc:.4f}")

# --- Metrics + confusion matrix
y_pred_prob = classifier.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=le.classes_))
print(f"\nDeep Learning Accuracy: {accuracy_score(y_true, y_pred):.4f}")

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Multi-class Confusion Matrix (Balanced Data)")
plt.xlabel("Predicted Disease")
plt.ylabel("True Disease")
plt.tight_layout()
plt.savefig(f"{FOLDER}/multiclass_confusion_matrix_balanced.png", dpi=150)
plt.close()

# --- Random Forest ensemble + robust SHAP for feature importance, per class
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, np.argmax(y_train, axis=1))
rf_score = rf.score(X_test, y_true)
print(f"\nRandomForest Ensemble Test Accuracy: {rf_score:.4f}")

explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# SHAP for each class, shape robust for multiclass RF
for class_idx, class_name in enumerate(le.classes_):
    curr_shap = shap_values.values[:,:,class_idx]  # shape (n_samples, n_features)
    feature_names_plot = shared_genes[:curr_shap.shape[1]]
    shap.summary_plot(curr_shap, X_test, feature_names=feature_names_plot, show=False)
    plt.title(f"SHAP Feature Importance for {class_name}")
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/shap_importance_{class_name}.png", dpi=150)
    plt.close()
