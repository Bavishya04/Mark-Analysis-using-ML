import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Read data
df = pd.read_csv("student_marks.csv")

# Let's create a target for simplicity (you can improve later)
# Rule: CGPA2 < 6.5 → At Risk, 6.5-8 → Safe, >8 → Topper
def get_category(cgpa):
    if cgpa < 6.5:
        return "At Risk"
    elif cgpa < 8.0:
        return "Safe"
    else:
        return "Topper"

df["Target"] = df["CGPA2"].apply(get_category)

# Features and target
X = df[["Math1", "Physics1", "Chem1", "CGPA1", "Math2", "Physics2", "Chem2", "CGPA2"]]
y = df["Target"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("✅ Model trained and saved as student_model.pkl")
