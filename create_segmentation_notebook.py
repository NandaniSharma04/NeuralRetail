import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Customer Segmentation (RFM + AI Clustering)\n",
    "This notebook categorizes customers based on their purchasing behavior using RFM (Recency, Frequency, Monetary) analysis.\n",
    "It uses 3 machine learning algorithms:\n",
    "1. **K-Means**: Core clustering into distinct segments.\n",
    "2. **DBSCAN**: Outlier detection (finding ultra-VIPs or anomalies).\n",
    "3. **Gaussian Mixture Models (GMM)**: Probabilistic clustering for borderline customers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required libraries (uncomment if running in a fresh environment)\n",
    "# !pip install pandas numpy scikit-learn matplotlib seaborn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans, DBSCAN\n",
    "from sklearn.mixture import GaussianMixture\n",
    "from sklearn.metrics import silhouette_score, davies_bouldin_score\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading & RFM Calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to the data file\n",
    "DATA_PATH = \"../data/new_cleaned_retail_data_with_churn.csv\"\n",
    "\n",
    "def calculate_rfm(filepath):\n",
    "    print(\"Loading data...\")\n",
    "    df = pd.read_csv(filepath)\n",
    "    \n",
    "    # Ensure invoicedate is datetime\n",
    "    df['invoicedate'] = pd.to_datetime(df['invoicedate'])\n",
    "    \n",
    "    # For Recency, we set a \"current date\" as 1 day after the last purchase in the dataset\n",
    "    current_date = df['invoicedate'].max() + pd.Timedelta(days=1)\n",
    "    \n",
    "    print(\"Calculating RFM metrics per customer...\")\n",
    "    # Aggregate by customer_id\n",
    "    # Recency: days since last purchase\n",
    "    # Frequency: number of unique invoices\n",
    "    # Monetary: total amount spent\n",
    "    rfm = df.groupby('customer_id').agg({\n",
    "        'invoicedate': lambda date: (current_date - date.max()).days,\n",
    "        'invoice': 'nunique',\n",
    "        'total_amount': 'sum'\n",
    "    }).reset_index()\n",
    "    \n",
    "    # Rename columns\n",
    "    rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']\n",
    "    \n",
    "    # Remove customers with 0 or negative monetary value to avoid skewing clusters\n",
    "    rfm = rfm[rfm['Monetary'] > 0]\n",
    "    \n",
    "    return rfm\n",
    "\n",
    "rfm_df = calculate_rfm(DATA_PATH)\n",
    "print(f\"Total customers analyzed: {len(rfm_df)}\")\n",
    "rfm_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Preprocessing (Scaling & Log Transformation)\n",
    "RFM data is often highly skewed (e.g., a few customers buy 100x more than others). We use Log Transformation to handle skewness, and StandardScaler to bring all 3 metrics to the same scale."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Log Transformation to handle skewness (add small epsilon to avoid log(0))\n",
    "rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()\n",
    "rfm_log['Recency'] = np.log1p(rfm_log['Recency'])\n",
    "rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])\n",
    "rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])\n",
    "\n",
    "# Standardization\n",
    "scaler = StandardScaler()\n",
    "rfm_scaled = scaler.fit_transform(rfm_log)\n",
    "\n",
    "rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])\n",
    "rfm_scaled_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. K-Means Clustering (Core Segmentation)\n",
    "We evaluate `k` (number of segments) from 6 to 10 using Silhouette Score and Davies-Bouldin Index to find the mathematical optimum."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_k = 6\n",
    "best_silhouette = -1\n",
    "best_kmeans_model = None\n",
    "\n",
    "print(\"Evaluating K-Means for k=6 to k=10...\")\n",
    "for k in range(6, 11):\n",
    "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "    cluster_labels = kmeans.fit_predict(rfm_scaled)\n",
    "    \n",
    "    # Calculate metrics\n",
    "    silhouette = silhouette_score(rfm_scaled, cluster_labels)\n",
    "    db_index = davies_bouldin_score(rfm_scaled, cluster_labels)\n",
    "    \n",
    "    print(f\"k={k} | Silhouette Score: {silhouette:.4f} | Davies-Bouldin: {db_index:.4f}\")\n",
    "    \n",
    "    # We want max Silhouette score\n",
    "    if silhouette > best_silhouette:\n",
    "        best_silhouette = silhouette\n",
    "        best_k = k\n",
    "        best_kmeans_model = kmeans\n",
    "\n",
    "print(f\"\\nOptimal number of segments selected: {best_k}\")\n",
    "\n",
    "# Apply best model\n",
    "rfm_df['Cluster_KMeans'] = best_kmeans_model.predict(rfm_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Assigning Business Personas to Clusters\n",
    "We look at the average RFM values for each cluster to label them properly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze average RFM per cluster\n",
    "cluster_summary = rfm_df.groupby('Cluster_KMeans').agg({\n",
    "    'Recency': 'mean',\n",
    "    'Frequency': 'mean',\n",
    "    'Monetary': 'mean',\n",
    "    'customer_id': 'count'\n",
    "}).rename(columns={'customer_id': 'Count'}).round(2)\n",
    "\n",
    "# Basic logic to assign personas based on relative RFM values\n",
    "def assign_persona(row):\n",
    "    if row['Recency'] < 30 and row['Frequency'] > 5 and row['Monetary'] > 2000:\n",
    "        return \"Champions\"\n",
    "    elif row['Recency'] > 180 and row['Frequency'] <= 2:\n",
    "        return \"Lost / Hibernating\"\n",
    "    elif row['Recency'] > 90 and row['Frequency'] > 5:\n",
    "        return \"At Risk (Loyal but inactive)\"\n",
    "    elif row['Recency'] < 60 and row['Frequency'] <= 2:\n",
    "        return \"Recent Customers\"\n",
    "    else:\n",
    "        return \"Average Customers\"\n",
    "\n",
    "cluster_summary['Persona_Label'] = cluster_summary.apply(assign_persona, axis=1)\n",
    "print(\"Cluster Personas:\")\n",
    "display(cluster_summary)\n",
    "\n",
    "# Map personas back to main dataframe\n",
    "persona_map = cluster_summary['Persona_Label'].to_dict()\n",
    "rfm_df['Persona'] = rfm_df['Cluster_KMeans'].map(persona_map)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. DBSCAN (Outlier & VIP Detection)\n",
    "DBSCAN groups dense data points together. Anything that doesn't fit closely into a cluster is labeled as `-1` (an Outlier). In retail, outliers with high spend are often our Ultra-VIPs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dbscan = DBSCAN(eps=0.5, min_samples=10)\n",
    "rfm_df['Cluster_DBSCAN'] = dbscan.fit_predict(rfm_scaled)\n",
    "\n",
    "outliers = rfm_df[rfm_df['Cluster_DBSCAN'] == -1]\n",
    "print(f\"DBSCAN found {len(outliers)} outliers/anomalies.\")\n",
    "print(\"Top 5 outliers by Monetary Value (Ultra-VIPs or unusual bulk buyers):\")\n",
    "display(outliers.sort_values(by='Monetary', ascending=False).head(5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Gaussian Mixture Models (Probabilistic Segmentation)\n",
    "Unlike K-Means (which says 'Customer X is 100% in Segment 2'), GMM calculates probabilities (e.g. 'Customer X is 80% Segment 2, and 20% Segment 3'). This is powerful for borderline customers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "gmm = GaussianMixture(n_components=best_k, random_state=42)\n",
    "gmm.fit(rfm_scaled)\n",
    "\n",
    "# Get the cluster assignment\n",
    "rfm_df['Cluster_GMM'] = gmm.predict(rfm_scaled)\n",
    "\n",
    "# Get the probabilities for each cluster\n",
    "probabilities = gmm.predict_proba(rfm_scaled)\n",
    "\n",
    "# Find max probability (how confident is the model?)\n",
    "rfm_df['GMM_Confidence'] = probabilities.max(axis=1)\n",
    "\n",
    "print(\"Top 5 Borderline Customers (Low Confidence in their segment):\")\n",
    "display(rfm_df.sort_values(by='GMM_Confidence').head(5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Export Results to Database/CRM\n",
    "The final dataframe now contains the customer personas and RFM scores. This is ready to be exported for the CRM system so Marketing can target them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_export = rfm_df[['customer_id', 'Recency', 'Frequency', 'Monetary', 'Persona']]\n",
    "print(\"Ready for CRM export:\")\n",
    "display(final_export.head())\n",
    "\n",
    "# Example export\n",
    "# final_export.to_csv(\"../data/customer_segments_export.csv\", index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("customer_segmentation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully: customer_segmentation.ipynb")
