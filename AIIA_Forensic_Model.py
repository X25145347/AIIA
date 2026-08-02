import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from joblib import dump
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import AIIA_Common as aiia_com

metadata_master_rows = []
metadata_master_rows = aiia_com.get_images_metadata("real")
metadata_master_rows.extend(aiia_com.get_images_metadata("ai"))

df = pd.DataFrame(metadata_master_rows)

categorical_cols = ["mode", "icc_present", "has_iCCP", "has_tEXt"]
numeric_cols = ["width", "height", "bytes_per_pixel", "num_qtables", "num_chunks", "qt_mean_0", "qt_std_0"]

# Build preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(
            handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Build full model pipeline
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

X = df[categorical_cols + numeric_cols]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1:", f1_score(y_val, y_pred, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
precision = precision_score(y_val, y_pred, average="macro")
recall = recall_score(y_val, y_pred, average="macro")

print("Precision:", precision)
print("Recall:", recall)
# Convert string labels to numeric
y_val_numeric = (y_val == "ai").astype(int)
y_pred_numeric = (y_pred == "ai").astype(int)
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_val_numeric, y_pred_numeric)

# Compute AUC
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Random Forest ROC curve (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Metadata Random Forest Classifier')
plt.legend(loc="lower right")
plt.savefig("roc_curve_metadata.png", dpi=300, bbox_inches='tight')

dump(clf, "metadata_forensic_model.joblib")