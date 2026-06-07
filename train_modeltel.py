"""
train_modeltel.py — No SMOTE version (uses class_weight instead)
Compatible with scikit-learn 1.9.0 + Python 3.14
"""

import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, roc_curve,
                                      precision_recall_curve, average_precision_score)
from xgboost import XGBClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & Clean ──────────────────────────────────────────────────────────────
df = pd.read_csv("TelcoCustomerChurn.csv")
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"]  = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ── Feature Engineering ───────────────────────────────────────────────────────
df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
df["TenureGroup"]      = pd.cut(
    df["tenure"], bins=[0,12,24,48,72],
    labels=[0,1,2,3], include_lowest=True
).astype("float").fillna(0).astype(int)
df["MultipleServices"] = sum(
    (df[c] == "Yes").astype(int)
    for c in ["OnlineSecurity","OnlineBackup","DeviceProtection",
              "TechSupport","StreamingTV","StreamingMovies"]
)

# ── Encode ────────────────────────────────────────────────────────────────────
cat_cols = df.select_dtypes(include=["object","str"]).columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Split ─────────────────────────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale + Impute ────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_s  = scaler.transform(imputer.transform(X_test))

print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print("Using class_weight='balanced' instead of SMOTE")

# ── 5 Models (class_weight handles imbalance) ─────────────────────────────────
models = {
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42,
        n_jobs=-1, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=0.5, random_state=42, class_weight="balanced"
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=42, class_weight="balanced"
    ),
}

results   = {}
best_auc  = 0
best_name = None
best_model= None
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Training {name}...", end=" ", flush=True)
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    cv_s    = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _= precision_recall_curve(y_test, y_proba)

    results[name] = {
        "model": model, "auc": auc, "ap": ap,
        "cv_mean": cv_s.mean(), "cv_std": cv_s.std(),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "precision": prec.tolist(), "recall": rec.tolist(),
        "y_proba": y_proba.tolist(), "y_test": y_test.tolist(),
    }
    safe = name.lower().replace(" ","_").replace("(","").replace(")","")
    joblib.dump(model, f"{MODEL_DIR}/model_{safe}.pkl")
    print(f"AUC={auc:.4f}  CV={cv_s.mean():.4f}±{cv_s.std():.4f}")
    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, model

print(f"\n✅ Best: {best_name} (AUC={best_auc:.4f})")

if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=feature_names)
elif hasattr(best_model, "coef_"):
    fi = pd.Series(np.abs(best_model.coef_[0]), index=feature_names)
else:
    lr = results["Logistic Regression"]["model"]
    fi = pd.Series(np.abs(lr.coef_[0]), index=feature_names)
fi = fi.sort_values(ascending=False)

model_fi = {}
for name, r in results.items():
    m = r["model"]
    if hasattr(m, "feature_importances_"):
        model_fi[name] = pd.Series(m.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif hasattr(m, "coef_"):
        model_fi[name] = pd.Series(np.abs(m.coef_[0]), index=feature_names).sort_values(ascending=False)
    else:
        lr = results["Logistic Regression"]["model"]
        model_fi[name] = pd.Series(np.abs(lr.coef_[0]), index=feature_names).sort_values(ascending=False)

joblib.dump(scaler,        f"{MODEL_DIR}/scaler.pkl")
joblib.dump(imputer,       f"{MODEL_DIR}/imputer.pkl")
joblib.dump(encoders,      f"{MODEL_DIR}/encoders.pkl")
joblib.dump(best_model,    f"{MODEL_DIR}/best_model.pkl")
joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")
joblib.dump(results,       f"{MODEL_DIR}/results.pkl")
joblib.dump(fi,            f"{MODEL_DIR}/feature_importance.pkl")
joblib.dump(best_name,     f"{MODEL_DIR}/best_model_name.pkl")
joblib.dump(model_fi,      f"{MODEL_DIR}/model_feature_importance.pkl")
print("All artifacts saved → ./models/")
