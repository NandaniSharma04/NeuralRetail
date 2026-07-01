# Demand Forecasting Code Explanation

Welcome to the world of Data Science and Machine Learning! This document is written specifically for beginners to help you understand exactly what the code does, piece by piece.

---

## 1. The Tech Stack (What are these tools?)

Here is a simple breakdown of the technologies used in this project:

- **Python 3.12**: The programming language used to write all the logic.
- **Pandas (`pd`)**: A library used for data manipulation. Think of it as Excel on steroids. It allows us to load CSV files into "DataFrames" (which are basically tables with rows and columns) and filter/aggregate data easily.
- **NumPy (`np`)**: A library for fast mathematical calculations and handling large arrays/lists of numbers.
- **Prophet**: An open-source library created by Facebook. It is specifically designed to forecast time-series data (like sales over time) and is very good at automatically handling weekly and yearly seasonality (e.g., higher sales on weekends or during Christmas).
- **PyTorch (`torch`)**: A Deep Learning framework created by Meta/Facebook. It allows us to build complex neural networks (like the human brain) to find hidden patterns in data.
- **PyTorch Lightning (`pl`)**: A wrapper around PyTorch that makes writing PyTorch code much easier and cleaner. It handles the complicated "training loops" and hardware stuff (like automatically using a GPU if one is available) so we don't have to write it ourselves.
- **LSTM (Long Short-Term Memory)**: A specific type of Neural Network architecture built inside PyTorch. It is designed to remember past events to predict future events, making it perfect for time-series forecasting.
- **MLflow**: A tool used to keep track of machine learning experiments. It logs how well a model performed (its accuracy) and saves the trained model so the backend team can load it later.
- **Scikit-learn / MinMaxScaler**: A machine learning library. We use its `MinMaxScaler` to shrink all our sales quantities to numbers between 0 and 1. Neural networks (like LSTM) learn much faster when the numbers are small.

---

## 2. What Does the Code Do? (High-Level Flow)

The goal of the notebook is to predict how many units of a product will be sold in the next 30 days. Here is the step-by-step process the code follows:

1. **Load Data**: It reads your large retail CSV file.
2. **Aggregate**: Instead of looking at individual invoices, it groups the data by Day and Product. For example, "On Dec 1st, we sold 50 units of Product A."
3. **Train Baseline (Prophet)**: For a given product, it asks Prophet to look at the history and guess the next 30 days. It then checks how accurate this guess was by comparing it to real past data (this accuracy score is called MAPE - Mean Absolute Percentage Error).
4. **Train Advanced Model (LSTM)**: If Prophet's error is higher than 10%, the code builds a deep learning model (LSTM). The LSTM looks at rolling 30-day "windows" of past sales to predict the 31st day.
5. **Combine & Pick the Best**: It compares Prophet's accuracy and LSTM's accuracy, and also tries combining their answers (an Ensemble). It picks whichever method had the lowest error.
6. **Save (MLflow)**: It saves the chosen model and its accuracy score into MLflow so your backend developer (Nandani) can use it.

---

## 3. Code Breakdown (Block by Block)

Here is a simplified explanation of the main code blocks in your Jupyter Notebook.

### Block 1: Imports
```python
import pandas as pd
from prophet import Prophet
import torch ...
```
**Meaning:** This is where we bring in external tools. Python doesn't know how to do deep learning by itself, so we `import` PyTorch and Prophet to give it those superpowers.

### Block 2: Data Preparation
```python
daily_sales = df.groupby(['date', 'stockcode'])['quantity'].sum().reset_index()
```
**Meaning:** This uses Pandas to take the raw invoice data, group it by the exact date and product ID (`stockcode`), and add up (`sum`) the quantities. We go from millions of individual transactions to a clean daily summary.

### Block 3: Prophet Training
```python
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(train)
forecast = model.predict(future)
```
**Meaning:** 
- We create a Prophet model and tell it to look out for weekly and yearly trends.
- `model.fit(train)` is where the actual "learning" happens. The model studies the historical data.
- `model.predict(future)` is where the model generates the forecast for the next 30 days.

### Block 4: LSTM Dataset Creation
```python
class TimeSeriesDataset(Dataset):
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])
```
**Meaning:** Deep Learning models can't just read dates. They need data formatted as "Inputs" (X) and "Answers" (Y). This block takes a long list of sales and chops it into sequences. For example, it takes Days 1-30 as the input (`x`) and Day 31 as the answer (`y`) that the model needs to guess.

### Block 5: The LSTM Brain
```python
class LSTMForecaster(pl.LightningModule):
    def __init__(self):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, 1)
```
**Meaning:** This defines the architecture of the neural network. The `LSTM` layer processes the timeline of data and finds patterns. The `Linear` layer takes those patterns and squeezes them down into a single number: the predicted sales quantity.

### Block 6: MLflow Logging
```python
with mlflow.start_run(run_name=f"Product_{product_id}"):
    mlflow.log_metric("prophet_mape", p_mape)
```
**Meaning:** MLflow acts like a diary. `start_run` creates a new page in the diary for a specific product. `log_metric` writes down the error percentage (MAPE) so we can look at a dashboard later and see exactly how well the model performed.

---

### Understanding the Inference Script (`inference.py`)
You also have an `inference.py` file. While the notebook is meant for **Training** (teaching the model), the inference script is meant for **Predicting** (using the model in the real world).

Nandani will use `inference.py` to:
1. Ping MLflow to grab the pre-trained brain for a specific product.
2. Feed it the current date.
3. Get back a clean list of forecasted quantities for the next 30 days and send it to your app's frontend.
