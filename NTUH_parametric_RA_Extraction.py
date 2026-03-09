import os
import nibabel as nib
import numpy as np
import pandas as pd

# Define paths
input_root = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB/realdPET_parametric_images"
roi_root = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB/ROI_masks"
output_excel_path = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB_RegionalActivity/realdPET_parametric_images/parametric_activity_by_region.xlsx"
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

# ROI mapping (match to 4D index in the mask)
roi_mapping = {
    "Frontal Lobe": 0,
    "Posterior Cingulate Cortex": 1,
    "Parietal Lobe": 2,
    "Temporal Lobe": 4,
    "Hippocampus + Amygdala": 10
}

# PET parameter images
pet_types = ["BPnd.nii", "R1.nii", "k2.nii"]

# Prepare output table dictionary
output_tables = {ptype.split(".")[0]: [] for ptype in pet_types}

# Loop through each patient folder
for patient_id in sorted(os.listdir(input_root)):
    patient_path = os.path.join(input_root, patient_id)
    if not os.path.isdir(patient_path):
        continue

    # Load this patient's 4D ROI mask
    mask_path = os.path.join(roi_root, patient_id, "binary_masks.nii")
    if not os.path.exists(mask_path):
        print(f"❌ Missing ROI mask for {patient_id}, skipping.")
        continue

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    for ptype in pet_types:
        ptype_name = ptype.split(".")[0]
        pet_path = os.path.join(patient_path, ptype)

        if not os.path.exists(pet_path):
            print(f"⚠️  {ptype} missing for {patient_id}, skipping.")
            continue

        # Load parametric PET data
        pet_data = nib.load(pet_path).get_fdata()

        # Extract mean intensity per ROI using 4D masks
        row = {"PatientID": patient_id}
        for roi_name, idx in roi_mapping.items():
            if idx >= mask_data.shape[3]:
                print(f"⚠️  {patient_id}: ROI index {idx} out of bounds, skipping {roi_name}")
                row[roi_name] = np.nan
                continue

            roi_mask = mask_data[:, :, :, idx]
            mask_voxels = roi_mask > 0

            if np.sum(mask_voxels) == 0:
                row[roi_name] = np.nan
                continue

            weighted_sum = np.sum(pet_data[mask_voxels] * roi_mask[mask_voxels])
            normalization = np.sum(roi_mask[mask_voxels])
            regional_value = weighted_sum / normalization

            row[roi_name] = regional_value

        output_tables[ptype_name].append(row)

# Save to Excel (one sheet per parametric image type)
with pd.ExcelWriter(output_excel_path) as writer:
    for ptype_name, rows in output_tables.items():
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=ptype_name[:31], index=False)

print(f"\n✅ Done! ROI values saved to: {output_excel_path}")
