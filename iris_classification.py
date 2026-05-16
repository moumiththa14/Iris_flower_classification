import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("IRIS.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop Id column if exists
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Standardize column names (VERY IMPORTANT)
df.columns = df.columns.str.lower()

print(df.head())
print(df.info())

# Missing values check (FIXED)
print(df.isnull().sum())

# Visualization
sns.pairplot(df, hue="species")
plt.show()

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: - iris_classification.py:58", accuracy * 100)

# Report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

predicted_species = encoder.inverse_transform(prediction)
print("Predicted Flower Species: - iris_classification.py:78", predicted_species[0])