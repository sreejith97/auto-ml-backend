import os
import urllib.request
import pandas as pd
import numpy as np
import json

# Set random seed for reproducibility
np.random.seed(42)

def generate_pima():
    print("Downloading Pima Indians Diabetes Dataset...")
    # Source URL: Jason Brownlee's canonical mirror of the UCI dataset
    pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    pima_file = "pima.csv"
    if not os.path.exists(pima_file):
        urllib.request.urlretrieve(pima_url, pima_file)
    
    # Original Columns
    # 0: Pregnancies
    # 1: Glucose
    # 2: BloodPressure
    # 3: SkinThickness
    # 4: Insulin
    # 5: BMI
    # 6: DiabetesPedigreeFunction
    # 7: Age
    # 8: Outcome (Target)
    df = pd.read_csv(pima_file, header=None)
    df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    
    ground_truth = []
    
    # 1. Missing Values (Pima naturally uses 0 for missing in certain columns)
    # We will explicitly convert some of these 0s to np.nan to test the pipeline's imputation
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]
    for col in missing_cols:
        zero_indices = df[df[col] == 0].index
        # Convert 50% of 0s to np.nan
        selected = np.random.choice(zero_indices, int(len(zero_indices)*0.5), replace=False)
        df.loc[selected, col] = np.nan
        ground_truth.append({
            "issue": "missing_values",
            "column": col,
            "original_0_indices": selected.tolist(),
            "expected_action": "impute_median" # As these are highly skewed numeric
        })
        
    # 2. Add a synthetic categorical column for casing tests
    # Documented for thesis: 'BloodType' is SYNTHETICALLY added.
    df["BloodType"] = np.random.choice(["A", "B", "AB", "O"], size=len(df))
    # Inject casing issues: Make 5% lowercase
    casing_idx = np.random.choice(df.index, int(len(df)*0.05), replace=False)
    df.loc[casing_idx, "BloodType"] = df.loc[casing_idx, "BloodType"].str.lower()
    
    ground_truth.append({
        "issue": "format_inconsistency",
        "column": "BloodType",
        "affected_indices": casing_idx.tolist(),
        "expected_action": "lowercase_all"
    })
    
    # 3. Inject Duplicate Rows (5%)
    dup_rows = df.sample(int(len(df)*0.05))
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    ground_truth.append({
        "issue": "duplicate_rows",
        "column": "Entire Dataset",
        "count": len(dup_rows),
        "expected_action": "drop_duplicates"
    })
    
    df.to_csv("pima_perturbed.csv", index=False)
    with open("pima_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Pima dataset perturbed. Shape: {df.shape}")

def generate_secom():
    print("Downloading SECOM Dataset...")
    # Source URL: UCI Machine Learning Repository
    secom_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    secom_file = "secom.data"
    
    if not os.path.exists(secom_file):
        # We need a custom header to bypass 403 Forbidden on UCI sometimes
        req = urllib.request.Request(secom_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(secom_file, 'wb') as out_file:
            out_file.write(response.read())
            
    df = pd.read_csv(secom_file, sep=" ", header=None)
    # The dataset has 590 features. We will just use the first 20 to keep tests fast, plus target
    # Wait, the target is in secom_labels.data. For simplicity of testing outliers, we just need the features.
    df = df.iloc[:, :20]
    df.columns = [f"sensor_{i}" for i in range(20)]
    
    ground_truth = []
    
    # 1. Inject outlier spikes (Reset artifacts) in 'sensor_0' and 'sensor_1'
    for col in ["sensor_0", "sensor_1"]:
        mean = df[col].mean()
        std = df[col].std()
        
        # Inject 3 spikes at known indices
        spike_indices = np.random.choice(df.index, 3, replace=False)
        df.loc[spike_indices, col] = mean + (10 * std) # Massive spike
        
        ground_truth.append({
            "issue": "outliers",
            "column": col,
            "spike_indices": spike_indices.tolist(),
            "expected_action": "cap_iqr"
        })
        
    df.to_csv("secom_perturbed.csv", index=False)
    with open("secom_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"SECOM dataset perturbed. Shape: {df.shape}")

if __name__ == "__main__":
    generate_pima()
    generate_secom()
