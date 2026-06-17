import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio

# Loading dataset
drug_df = pd.read_csv("Data/drug.csv")
drug_df.head(3)

# Train Test split
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=91
)

# Creating the pipelines
cat_col = [1, 2, 3]
num_col = [0, 4]

# Preprocessing pipeline
transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

# Training pipeline
pipe = Pipeline(
    steps=[
        ("prerpocessing", transform),
        ("model", RandomForestClassifier(n_estimators=80, random_state=12)),
    ]
)

pipe.fit(X_train, y_train)

# Model Evaluation
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy:, {accuracy*100}%, F1 score: {f1}")

# Save to metrics file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {accuracy}, F1 score: {f1}")

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
# Saving the model
sio.dump(pipe, "Model/drug_pipeline.skops")
