import os

append_text = """

---

# Part 2: Customer Segmentation (RFM & Clustering)

This section explains the logic inside `customer_segmentation.ipynb`.

## 1. The New Tech Stack (For Segmentation)

- **Scikit-Learn**: The most popular Python library for standard machine learning. We use it for scaling data and all our clustering algorithms.
- **Seaborn**: A visualization library based on matplotlib. It makes beautiful statistical graphs.

## 2. What Does the Code Do? (High-Level Flow)

1. **Calculate RFM**: We summarize each customer by 3 metrics:
   - **R (Recency)**: How many days since they last bought something?
   - **F (Frequency)**: How many separate orders have they placed?
   - **M (Monetary)**: How much money have they spent in total?
2. **K-Means Clustering**: The AI analyzes the RFM numbers and mathematically groups similar customers together into segments (like "VIPs" or "At-Risk").
3. **DBSCAN (Outlier Detection)**: Finds the weird data points that don't fit into normal groups. In retail, these are usually "Ultra-VIPs" who buy in massive bulk.
4. **Gaussian Mixture Models (GMM)**: Instead of a hard "Yes/No", this model gives us a *probability* (e.g., this customer is 70% likely to be in Segment A, and 30% likely to be in Segment B).

## 3. Key Concepts Explained Simply

### K-Means (How does it pick groups?)
Imagine spreading your customers on a giant map based on how much they spend. K-Means drops `k` pins on the map (let's say 6 pins). Every customer walks to the nearest pin. The pins then move to the center of their crowd. This repeats until the pins stop moving. The crowds around the pins are your "Clusters".

### Silhouette Score & Davies-Bouldin Index
How do we know if 6 pins or 10 pins is better? 
- **Silhouette Score**: Measures how tightly packed a crowd is, and how far away it is from other crowds. (Higher is better).
- **Davies-Bouldin**: Measures how "overlapping" the crowds are. (Lower is better).
Our code automatically tests 6, 7, 8, 9, and 10 pins and uses these scores to pick the absolute best number of clusters!

### Personas
Once the AI groups the customers, it just labels them `Cluster 0`, `Cluster 1`, etc. We wrote a piece of code (the `assign_persona` function) to look at the averages of each cluster and translate them into English labels (like "Champions" or "Lost/Hibernating").
"""

with open("Code_Explanation.md", "a", encoding="utf-8") as f:
    f.write(append_text)

print("Appended explanation.")
