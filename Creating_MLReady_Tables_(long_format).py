# script_merge_early_late_wide.py

import pandas as pd
import os

# === USER INPUTS: customize these paths ===
early_excel = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/ML_ready_tables/syn_ADNI_FDG/syn_ADNI_suvr_regional_activity_early.xlsx"
late_excel  = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/ML_ready_tables/syn_ADNI_FDG/syn_ADNI_suvr_regional_activity_late.xlsx"
output_dir  = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/Final_MLReady/syn_ADNI_FDG"

# Create output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# === Load datasets ===
early_df = pd.read_excel(early_excel)
late_df  = pd.read_excel(late_excel)

# Sanity check: make sure required columns exist
required_cols = {"Patient", "Frame", "Regional_Activity", "Voxel_Count", "ROI"}
for name, df in [("Early", early_df), ("Late", late_df)]:
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{name} file is missing one of the required columns: {required_cols}")

# === Rename columns to distinguish Early vs Late ===
early_df = early_df.rename(columns={
    "Frame": "Early_Frame",
    "Regional_Activity": "Early_Regional_Activity",
    "Voxel_Count": "Early_Voxel_Count"
})
late_df = late_df.rename(columns={
    "Frame": "Late_Frame",
    "Regional_Activity": "Late_Regional_Activity",
    "Voxel_Count": "Late_Voxel_Count"
})

# === Merge on Patient + ROI ===
merged = pd.merge(
    early_df[["Patient", "ROI", "Early_Frame", "Early_Regional_Activity", "Early_Voxel_Count"]],
    late_df[["Patient", "ROI", "Late_Frame", "Late_Regional_Activity", "Late_Voxel_Count"]],
    on=["Patient", "ROI"],
    how="outer"  # keep all patients even if one timepoint is missing
)

# === Sort columns: keep ID columns first, then Early, then Late ===
col_order = ["Patient", "ROI",
             "Early_Frame", "Early_Regional_Activity", "Early_Voxel_Count",
             "Late_Frame", "Late_Regional_Activity", "Late_Voxel_Count"]
merged = merged[col_order]

# === Save outputs ===
excel_out = os.path.join(output_dir, "ML_ready_dataset.xlsx")
csv_out   = os.path.join(output_dir, "ML_ready_dataset.csv")

merged.to_excel(excel_out, index=False)
merged.to_csv(csv_out, index=False)

print(f"✅ Wide-format dataset saved as:\n{excel_out}\n{csv_out}")
