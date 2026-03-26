# fraud_detection.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# ----------------------------
# 1. Charger dataset
# ----------------------------
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

print("Dataset shape:", data.shape)
print(data.head())

# ----------------------------
# 2. Target
# ----------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# ----------------------------
# 3. Split train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------
# 4. Modèle Random Forest
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 5. Prédictions
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ----------------------------
# 6. Évaluation
# ----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", roc_score)

# ----------------------------
# 7. Graphiques
# ----------------------------
# Confusion Matrix
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()
