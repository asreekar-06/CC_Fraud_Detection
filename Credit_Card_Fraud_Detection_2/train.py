# Run this script in Google Colab after uploading fraudTrain.csv and fraudTest.csv.
# It preserves the Claude notebook's method: stratified split, train-only SMOTE,
# compare LR / Linear SVM / Random Forest / XGBoost, validation review-band report,
# then save the Random Forest and thresholds for the web app.

import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (confusion_matrix, average_precision_score,
                             precision_recall_fscore_support)
from xgboost import XGBClassifier

DATA_DIR = "/content"
TRAIN_CSV = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST_CSV = os.path.join(DATA_DIR, "fraudTest.csv")
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "is_fraud"
NUMERIC = ["amt", "hour", "day_of_week"]
CATEGORICAL = ["category"]
FEATURES = NUMERIC + CATEGORICAL

def prepare_frame(df):
    df = df.copy()
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
    required = ["amt", "category", "hour", "day_of_week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce")
    df["category"] = df["category"].fillna("unknown").astype(str)
    df = df.dropna(subset=["amt", "hour", "day_of_week"])
    return df

print("Loading dataset files...")
train_raw = pd.read_csv(TRAIN_CSV)
test_raw = pd.read_csv(TEST_CSV)
train_raw = prepare_frame(train_raw)
test_raw = prepare_frame(test_raw)

# Use the dataset's supplied train/test files. The supplied test file remains untouched.
X_all = train_raw[FEATURES].copy()
y_all = pd.to_numeric(train_raw[TARGET], errors="raise").astype(int)
X_test = test_raw[FEATURES].copy()
y_test = pd.to_numeric(test_raw[TARGET], errors="raise").astype(int)

# Validation split is carved out of the training file only, as in Claude's notebook.
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ]), NUMERIC),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL)
])

# Fit preprocessing on the training portion only, then apply SMOTE to training rows only.
Xtr_enc = preprocessor.fit_transform(X_train)
Xval_enc = preprocessor.transform(X_val)
Xtest_enc = preprocessor.transform(X_test)
Xtr_enc = Xtr_enc.toarray() if hasattr(Xtr_enc, "toarray") else Xtr_enc
Xval_enc = Xval_enc.toarray() if hasattr(Xval_enc, "toarray") else Xval_enc
Xtest_enc = Xtest_enc.toarray() if hasattr(Xtest_enc, "toarray") else Xtest_enc

print("Training distribution before SMOTE:", y_train.value_counts().to_dict())
X_res, y_res = SMOTE(random_state=42).fit_resample(Xtr_enc, y_train)
print("Training distribution after SMOTE:", pd.Series(y_res).value_counts().to_dict())

def evaluate(model, name, Xtr, ytr, Xeval, yeval):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xeval)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        yeval, pred, average="binary", zero_division=0
    )
    print(f"{name}: precision={p:.3f} recall={r:.3f} F1={f:.3f} "
          f"PR-AUC={average_precision_score(yeval, proba):.3f}")
    print(confusion_matrix(yeval, pred))
    return proba

# Compare the same model families as Claude's notebook.
lr = LogisticRegression(max_iter=1000)
lr_proba = evaluate(lr, "Logistic Regression", X_res, y_res, Xtest_enc, y_test)

# Linear SVM is trained on a 100K sample, matching the original notebook's approach.
rng = np.random.RandomState(42)
sample_n = min(100000, len(X_res))
sample_idx = rng.choice(len(X_res), size=sample_n, replace=False)
svm = CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3)
svm_proba = evaluate(svm, "Linear SVM", X_res[sample_idx], np.asarray(y_res)[sample_idx],
                     Xtest_enc, y_test)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_proba = evaluate(rf, "Random Forest", X_res, y_res, Xtest_enc, y_test)

xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    tree_method="hist", n_jobs=-1, random_state=42,
                    eval_metric="logloss")
xgb_proba = evaluate(xgb, "XGBoost", X_res, y_res, Xtest_enc, y_test)

def band_report(y_true, proba, low, high):
    y = np.asarray(y_true)
    auto_fraud = proba >= high
    auto_ok = proba <= low
    review = ~(auto_fraud | auto_ok)
    return {
        "low": low, "high": high,
        "review_percent": round(100 * review.mean(), 2),
        "missed_frauds_auto_genuine": int(((y == 1) & auto_ok).sum()),
        "false_alarms_auto_fraud": int(((y == 0) & auto_fraud).sum()),
        "frauds_in_review": int(((y == 1) & review).sum()),
        "genuine_in_review": int(((y == 0) & review).sum())
    }

# Choose the same 0.2 / 0.8 band used by Claude's app.
low, high = 0.2, 0.8
rf_band = band_report(y_test, rf_proba, low, high)
print("Random Forest review-band report:", rf_band)

# Save the fitted preprocessing and model together, so API input processing is identical.
artifact = {
    "model": rf,
    "preprocessor": preprocessor,
    "features": FEATURES,
    "numeric": NUMERIC,
    "categorical": CATEGORICAL,
    "low": low,
    "high": high,
    "metrics": {
        "random_forest_pr_auc": float(average_precision_score(y_test, rf_proba)),
        "random_forest_band": rf_band
    }
}
joblib.dump(artifact, os.path.join(OUTPUT_DIR, "fraud_model.joblib"), compress=3)
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(artifact["metrics"], f, indent=2)

# Generate sample rows using readable fields for the frontend sample buttons.
samples = test_raw.sample(min(40, len(test_raw)), random_state=42).copy()
sample_out = pd.DataFrame({
    "amt": samples["amt"],
    "category": samples["category"],
    "hour": samples["hour"],
    "day_of_week": samples["day_of_week"],
    "is_fraud": samples[TARGET].astype(int)
})
sample_out.to_csv(os.path.join(OUTPUT_DIR, "sample_transactions.csv"), index=False)
print("Saved files to", OUTPUT_DIR)
print("Download fraud_model.joblib and sample_transactions.csv from Colab output folder.")
