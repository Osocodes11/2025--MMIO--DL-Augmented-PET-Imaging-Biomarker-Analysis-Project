import os
import nibabel as nib
import numpy as np
import pandas as pd

roi_root = '/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB/ROI_masks'
pet_root = '/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB/realdPET_suvr_images'
output_dir = '/home/mmiob11/MMIO_Open_Challenge_Summer_2025/NTUH_PiB_RegionalActivity/realdPET_suvr'
os.makedirs(output_dir, exist_ok=True)

# Use your defined ROI index mapping
roi_mapping = {
    "Frontal Lobe": 0,
    "Posterior Cingulate Cortex": 1,
    "Parietal Lobe": 2,
    "Temporal Lobe": 4,
    "Hippocampus + Amygdala": 10
}

frames_of_interest = {
    "Early": "frame06.nii",   # 3.0–4.0 min
    "Late": "frame21.nii"     # 40.0–50.0 min
}

# Create a dict to hold results per ROI
roi_tables = {roi_name: [] for roi_name in roi_mapping}

patients = sorted(os.listdir(roi_root))
for patient in patients:
    mask_path = os.path.join(roi_root, patient, 'binary_masks.nii')
    pet_folder = os.path.join(pet_root, patient)

    if not os.path.exists(mask_path):
        print(f"Missing mask for {patient}, skipping.")
        continue
    if not os.path.exists(pet_folder):
        print(f"Missing PET folder for {patient}, skipping.")
        continue

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    for timepoint, frame_file in frames_of_interest.items():
        frame_path = os.path.join(pet_folder, frame_file)
        if not os.path.exists(frame_path):
            print(f"{patient}: missing {frame_file}, skipping.")
            continue

        pet_data = nib.load(frame_path).get_fdata()

        for roi_name, idx in roi_mapping.items():
            roi_mask = mask_data[:, :, :, idx]
            roi_voxels = roi_mask > 0
            if np.sum(roi_voxels) == 0:
                continue

            weighted_sum = np.sum(pet_data[roi_voxels] * roi_mask[roi_voxels])
            mask_sum = np.sum(roi_mask[roi_voxels])
            regional_activity = weighted_sum / mask_sum

            roi_tables[roi_name].append({
                'Patient': patient,
                'Timepoint': timepoint,
                'Frame': frame_file.replace('.nii', ''),
                'Regional_Activity': regional_activity,
                'Voxel_Count': int(np.sum(roi_voxels))
            })

# Save one CSV file per ROI
for roi_name, table in roi_tables.items():
    df = pd.DataFrame(table)
    csv_path = os.path.join(output_dir, f"{roi_name.replace(' ', '_')}.csv")
    df.to_csv(csv_path, index=False)

# Save to a single Excel file with multiple sheets
excel_path = os.path.join(output_dir, "regional_activity_by_region.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    for roi_name, table in roi_tables.items():
        df = pd.DataFrame(table)
        df.to_excel(writer, sheet_name=roi_name[:31], index=False)  # Excel sheet names max = 31 chars

print("✅ Done! Output saved by ROI to both CSV and Excel files.")
