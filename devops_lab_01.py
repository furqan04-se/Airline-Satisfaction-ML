from google.colab import drive
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

# 1. Mount Drive & Load Data
drive.mount('/content/drive')
train = pd.read_csv('/content/drive/MyDrive/Dataset/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Dataset/test.csv')

# 2. Clean & Drop Unused Columns
cols_to_drop = [c for c in [train.columns[0], 'id'] if c in train.columns]
train = train.drop(columns=cols_to_drop).dropna()
test = test.drop(columns=cols_to_drop).dropna()

# 3. Select Features and Target
features = [
    'Online boarding', 'Inflight wifi service', 'Class', 'Type of Travel',
    'Inflight entertainment', 'Seat comfort', 'Leg room service',
    'Ease of Online booking', 'Customer Type', 'Flight Distance'
]
target = 'satisfaction'

X_train, y_train = train[features].copy(), train[target].copy()
X_test, y_test = test[features].copy(), test[target].copy()

# 4. Helper Function for Evaluation (DRY Principle)
def evaluate_model(model_name, y_true, y_pred):
    print(f"=== {model_name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("-" * 50)

# =========================================================
# PART A: Random Forest & Decision Tree
# =========================================================
cat_cols = ['Class', 'Type of Travel', 'Customer Type']
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    # Handle unseen test categories gracefully if needed
    X_test[col] = X_test[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
    X_test[col] = le.transform(X_test[col])

le_target = LabelEncoder()
y_train = y_train_encoded = le_target.fit_transform(y_train)
y_test = y_test_encoded = le_target.transform(y_test)

# Train Random Forest
rf = RandomForestClassifier(max_depth=7, n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_encoded)
evaluate_model("RANDOM FOREST", y_test_encoded, rf.predict(X_test))

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=15, random_state=42)
dt.fit(X_train, y_train_encoded)
evaluate_model("DECISION TREE", y_test_encoded, dt.predict(X_test))

# =========================================================
# PART B: Histogram Gradient Boosting Classifier (Native Categorical)
# =========================================================
print("==================================================")
print("    PART B - NEW MODEL: HISTOGRAM GRADIENT BOOSTING")
print("==================================================")

# Reload raw X data for HGB if you want to use native categorical handling, 
# or keep the encoded version. Here we use the encoded features cleanly:
hgb = HistGradientBoostingClassifier(random_state=42)
hgb.fit(X_train, y_train_encoded)

evaluate_model("HISTOGRAM GRADIENT BOOSTING", y_test_encoded, hgb.predict(X_test))
