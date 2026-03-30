import pandas as pd

df = pd.read_csv("Data/cleaned_diabetes.csv")

print(df.head())
print(df.columns)
print(df.shape)