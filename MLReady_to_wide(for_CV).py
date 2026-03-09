import pandas as pd
import os

def convert_long_to_wide(input_file, output_file):
    """
    Convert long-format ROI feature file into wide-format (per patient).

    Parameters:
        input_file (str): Path to the long-format Excel/CSV file
        output_file (str): Path to save the wide-format file (Excel/CSV)
    """

    # Load file
    df = pd.read_excel(input_file) if input_file.endswith(('.xls', '.xlsx')) else pd.read_csv(input_file)

    # Check columns
    expected_cols = ["Patient", "ROI", "Early_Regional_Activity", "Late_Regional_Activity"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in input file")

    # Pivot table (wide format)
    df_wide = df.pivot_table(
        index="Patient",
        columns="ROI",
        values=["Early_Regional_Activity", "Late_Regional_Activity"],
        aggfunc="first"
    )

    # Flatten MultiIndex column names
    df_wide.columns = [f"{roi}_{'early' if act.startswith('Early') else 'late'}"
                       for act, roi in df_wide.columns]

    # Reset index to keep Patient as a column
    df_wide = df_wide.reset_index()

    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save output
    if output_file.endswith(('.xls', '.xlsx')):
        df_wide.to_excel(output_file, index=False)
    else:
        df_wide.to_csv(output_file, index=False)

    print(f"✅ Converted file saved to {output_file}")


# --- Run when executed directly ---
if __name__ == "__main__":
    input_file = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/Final_MLReady/syn_NTUH_PiB/frame12/syn_NTUH_frame12_MLReady.xlsx"
    output_file = "/home/mmiob11/MMIO_Open_Challenge_Summer_2025/wide_Final_MLReady/syn_NTUH_FiB/frame12/syn_NTUH_frame12_wide_ML.xlsx"

    convert_long_to_wide(input_file, output_file)
