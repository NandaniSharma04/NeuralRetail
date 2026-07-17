import json

with open("customer_segmentation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        sources = cell.get("source", [])
        new_sources = []
        for line in sources:
            if 'DATA_PATH = "../data/new_cleaned_retail_data_with_churn.csv"' in line:
                line = line.replace('DATA_PATH = "../data/new_cleaned_retail_data_with_churn.csv"', 'DATA_PATH = "data/new_cleaned_retail_data_with_churn.csv"')
            new_sources.append(line)
        cell["source"] = new_sources

with open("customer_segmentation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
