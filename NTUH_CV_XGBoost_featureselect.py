# KFold_CV_XGBoost_with_metrics_and_importance.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report 

# ===============================
# User settings (EDIT THESE)
# ===============================
suvr_file       = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/wide_Final_MLReady/syn_NTUH_FiB/frame12/syn_NTUH_suvr_frame12_wide_ML.xlsx"
nonsuvr_file    = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/wide_Final_MLReady/syn_NTUH_FiB/frame12/syn_NTUH_frame12_wide_ML.xlsx"
parametric_file = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB_RegionalActivity/synthetic_realdPET/syndPET_parametric/frame05/syn_NTUH_param_regional_activity_early.xlsx"
diagnosis_file  = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB/NTUH_PiB_metadata(DrTsaiLiu_checked).xlsx"

output_file     = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/CV_Results/synthetic_realdPET/syn_NTUH/fselect_CV_syn_NTUH_frame12_XGBoost.xlsx"
n_splits        = 5
random_state    = 42
diagnosis_sheet = 0  # change if needed

# ===============================
# Helpers
# ===============================
def _resolve(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")
    return p

def _standardize_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize likely ID column names to 'PatientID'
    cols = {c.strip(): c for c in df.columns}  # map stripped->original
    candidates = ["PatientID", "Patient", "ID", "Subject", "Unnamed: 0"]
    found = next((cols[c] for c in candidates if c in cols), None)
    if found is None:
        # If the first column is unnamed like 'Unnamed: 0' or blank, use it.
        first = df.columns[0]
        if str(first).startswith("Unnamed"):
            df = df.rename(columns={first: "PatientID"})
        else:
            raise KeyError("No PatientID/Patient/ID/Subject/Unnamed: 0 column found.")
    else:
        if found != "PatientID":
            df = df.rename(columns={found: "PatientID"})
    # Clean up ID values (strings with trimmed spaces)
    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    return df

def load_single_sheet(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(_resolve(filepath), sheet_name=0)
    return _standardize_patient_id(df)

def load_parametric(filepath: str) -> pd.DataFrame:
    xls = pd.ExcelFile(_resolve(filepath))
    combined = pd.DataFrame()
    for sheet in xls.sheet_names:
        d = pd.read_excel(xls, sheet_name=sheet)
        d = _standardize_patient_id(d)
        # prefix all non-ID columns with sheet name to avoid collisions
        d = d.rename(columns={c: f"param_{sheet}_{c}" for c in d.columns if c != "PatientID"})
        combined = d if combined.empty else pd.merge(combined, d, on="PatientID", how="inner")
    return combined

# ===============================
# Load data
# ===============================
print("Loading SUVR & non-SUVR...")
df_suvr    = load_single_sheet(suvr_file)
df_nonsuvr = load_single_sheet(nonsuvr_file)

print("SUVR file columns (first 10):", df_suvr.columns[:10])
print("Non-SUVR file columns (first 10):", df_nonsuvr.columns[:10])

print("Loading parametric (multi-sheet)...")
df_parametric = load_parametric(parametric_file)

print("Loading diagnosis...")

# Step 1: Inspect sheet names
xls = pd.ExcelFile(_resolve(diagnosis_file))
print("📑 Available sheets in diagnosis file:", xls.sheet_names)

# Step 2: Load the chosen sheet
df_diag = pd.read_excel(
    xls,
    sheet_name=diagnosis_sheet,  # still using your variable (can be int or str)
    header=1,  # <-- may need to adjust later depending on preview
    keep_default_na=False,
    na_values=[]
)

print("🔎 Preview of loaded diagnosis sheet (first 5 rows):")
print(df_diag.head())
print("🔎 Columns found:", df_diag.columns.tolist())

# Rename first column to PatientID (if appropriate)
df_diag.rename(columns={df_diag.columns[0]: "PatientID"}, inplace=True)

# Keep only PatientID + Diagnosis if present
if "Diagnosis" in df_diag.columns:
    df_diag = df_diag[["PatientID", "Diagnosis"]]
else:
    print("⚠️ Column 'Diagnosis' not found yet. Check preview above and adjust header/col names.")

# Drop empty rows
df_diag = df_diag.dropna(how="all")

# Standardize IDs
df_diag = _standardize_patient_id(df_diag)

# Rename first column to PatientID
df_diag.rename(columns={df_diag.columns[0]: "PatientID"}, inplace=True)

# Keep only PatientID and Diagnosis
df_diag = df_diag[["PatientID", "Diagnosis"]]

# Drop any completely empty rows
df_diag = df_diag.dropna(how="all")

# Standardize IDs
df_diag = _standardize_patient_id(df_diag)

# ===============================
# Merge all datasets
# ===============================
print("Merging datasets on PatientID...")
dfs = [df_suvr, df_nonsuvr, df_parametric]
df_merged = df_diag
for d in dfs:
    df_merged = pd.merge(df_merged, d, on="PatientID", how="inner")

    print("Merged columns:", df_merged.columns[:20])

print(f"✅ Final merged shape: {df_merged.shape}")
print("🔎 Checking class balance...")
print(df_merged['Diagnosis'].value_counts())
print(df_merged['Diagnosis'].value_counts(normalize=True))

# ===============================
# Prepare X, y
# ===============================
if "Diagnosis" not in df_merged.columns:
    raise KeyError("Column 'Diagnosis' not found in diagnosis file after merge.")
X = df_merged.drop(columns=["PatientID", "Diagnosis"])
y = df_merged["Diagnosis"]

# Encode diagnosis labels
le = LabelEncoder()
y = pd.Series(
    le.fit_transform(y),
    index=X.index,
    name="Diagnosis"
)

# Print label mapping (debug)
print("🔎 Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# --- Save merged dataset for inspection ---
output_debug = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/Codes/debug_merged_dataset.xlsx"
df_merged.to_excel(output_debug, index=False)
print(f"🔎 Debug: merged dataset exported to {output_debug}")
print(f"✅ Final merged shape: {df_merged.shape}")

# ===============================
# Step 1: First CV run to get feature importance
# ===============================
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
feat_importances = pd.DataFrame(index=X.columns)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_tr, y_tr)

    # Store feature importance for this fold
    feat_importances[f"Fold{fold}"] = model.feature_importances_

# Compute mean importance and cumulative
feat_importances["Mean_Importance"] = feat_importances.mean(axis=1)
feat_importances = feat_importances.sort_values("Mean_Importance", ascending=False)
feat_importances["Cumulative"] = feat_importances["Mean_Importance"].cumsum() / feat_importances["Mean_Importance"].sum()

# ===============================
# Step 2: Filter top 90% features
# ===============================
top_features = feat_importances[feat_importances["Cumulative"] <= 0.9].index.tolist()
X_filtered = X[top_features]

# Optional: save filtered dataset
df_filtered_debug = pd.concat([df_merged[["PatientID", "Diagnosis"]], X_filtered], axis=1)
df_filtered_debug.to_excel("/home/mmiob11/MMIO_Open_Challenge_Summer_2025/Debug_Datasets/debug_filtered_dataset.xlsx", index=False)
print(f"🔎 Debug: filtered dataset exported to debug_filtered_dataset.xlsx")

# ===============================
# Step 3: Run CV on filtered features
# ===============================
per_fold = []
feat_importances_filtered = pd.DataFrame(index=X_filtered.columns)
cms = []
reports = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_filtered, y), start=1):
    X_tr, X_te = X_filtered.iloc[train_idx], X_filtered.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)

    # Metrics
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    try:
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_te, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = np.nan

    per_fold.append(
        {"Fold": fold, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}
    )

    # Store feature importance for this fold
    feat_importances_filtered[f"Fold{fold}"] = model.feature_importances_

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=range(len(le.classes_)))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in le.classes_],
                         columns=[f"Pred_{c}" for c in le.classes_])
    cm_df.insert(0, "Fold", fold)
    cms.append(cm_df)

    # Classification report
    report = classification_report(y_te, y_pred,
                                   labels=range(len(le.classes_)),
                                   target_names=le.classes_,
                                   output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.insert(0, "Fold", fold)
    reports.append(report_df)

# Build results DataFrames
results_df = pd.DataFrame(per_fold)
mean_row = results_df.mean(numeric_only=True).to_dict()
mean_row["Fold"] = "Mean"
results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

feat_importances_filtered["Mean_Importance"] = feat_importances_filtered.mean(axis=1)
feat_importances_filtered = feat_importances_filtered.sort_values("Mean_Importance", ascending=False)

# ===============================
# Save to Excel
# ===============================
print(f"Saving results to: {output_file}")
df_cms = pd.concat(cms)
df_reports = pd.concat(reports)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="CV_Results", index=False)
    feat_importances_filtered.reset_index(names="Feature").to_excel(writer, sheet_name="Feature_Importance", index=False)
    df_cms.to_excel(writer, sheet_name="Confusion_Matrices", index=False)
    df_reports.to_excel(writer, sheet_name="Classification_Reports")

print("Done ✅")

