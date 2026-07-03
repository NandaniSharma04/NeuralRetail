import json
import os

notebooks = [
    "Demand_Forecasting/demand_forecasting.ipynb",
    "Customer_Segmentation/customer_segmentation.ipynb",
    "Churn_Prediction/churn_prediction.ipynb",
    "Churn_Prediction/churn_prediction-checkpoint.ipynb"
]

target_path = '"../data/raw/new_cleaned_retail_data_with_churn.csv"'

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"Skipping {nb_path} - not found.")
        continue
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            sources = cell.get("source", [])
            new_sources = []
            for line in sources:
                if 'DATA_PATH =' in line and '.csv"' in line:
                    new_line = f'DATA_PATH = {target_path}\n'
                    if line != new_line:
                        line = new_line
                        changed = True
                new_sources.append(line)
            cell["source"] = new_sources

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Updated paths in {nb_path}")
    else:
        print(f"No changes needed in {nb_path}")
