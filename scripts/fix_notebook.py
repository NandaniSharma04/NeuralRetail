import json

with open("demand_forecasting.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        sources = cell.get("source", [])
        new_sources = []
        for line in sources:
            if "return torch.FloatTensor(x).unsqueeze(1)" in line:
                line = line.replace("return torch.FloatTensor(x).unsqueeze(1)", "return torch.FloatTensor(x)")
            new_sources.append(line)
        cell["source"] = new_sources

with open("demand_forecasting.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
