# fraud_detection.py
# Détection de fraude bancaire avec Random Forest + visualisation
# Version Colab-ready (dataset chargé depuis URL)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Charger les données depuis internet
# ---------------------------
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

print("Dataset loaded. Shape:", data.shape)
print(data.head())

# ---------------------------
# Analyse rapide
# ---------------------------
# Vérifier proportion de fraude
fraud_ratio = data['Class'].value_counts(normalize=True)
print("\nProportion de transactions frauduleuses :")
print(fraud_ratio)

# Graphique stylé
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)
plt.title("Distribution Transactions Fraud / Non-Fraud")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# ---------------------------
# Préparer features et target
# ---------------------------
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Modèle Random Forest
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Prédictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Évaluation
# ---------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix graphique stylé
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------
# Feature importance
# ---------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 10 Features Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
