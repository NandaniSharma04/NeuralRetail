import pandas as pd
import numpy as np
import os

input_file = "data/online_retail_II.xlsx"
output_file = "data/new_cleaned_retail_data.csv"

print("Loading Dataset...")
df= pd.read_excel(input_file)

print("Original Shape:")
print(df.shape)

#Column Cleaning\\
df.columns = (df.columns.str.strip().str.lower().str.replace(" ","_"))

print(df.columns)

df.drop_duplicates(inplace= True)
print("After removing duplicates:")
print(df.shape)

#Handle missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df.dropna(subset=["customer_id"],inplace=True)


df["description"].fillna("Unknown Product",inplace=True)

df["price"]=(df["price"].fillna(df["price"].median()))

df["quantity"]=(df["quantity"].fillna(df["quantity"].median()))

df["country"] = (df["country"].fillna(df["country"].mode()[0]))

df["invoicedate"] = df["invoicedate"].dt.strftime("%Y-%m-%d %H:%M:%S")



df["customer_id"] = (df["customer_id"].astype(int))

df = df[df["quantity"] > 0] #quantity can't be zero
df = df[df["price"]>0]

df["total_amount"] = (df["quantity"] * df["price"])


df["invoicedate"] = pd.to_datetime(
    df["invoicedate"]
)

df["year"] = (df["invoicedate"].dt.year)
df["month"] = (df["invoicedate"].dt.month)

df["day"] = (df["invoicedate"].dt.day)
df["day_of_week"] = (df["invoicedate"].dt.day_name())


df.sort_values(by="invoicedate",inplace=True)

q1= df["quantity"].quantile(0.25)
q3 = df["quantity"].quantile(0.75)

iqr = q3-q1
lower = q1- 1.5*iqr
upper = q3 + 1.5*iqr

df= df[(df["quantity"]>= lower) & (df["quantity"])<=upper]


print("\nMissing values after cleaning:")
print(df.isnull().sum())

#df.to_excel("output_file",index=False)

# saving the data!
df.to_csv(output_file,index=False)

print("\nCleaning Completed")
print("Final Shape:")
print(df.shape)


print("\nSaved:")
print(output_file)