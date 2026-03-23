from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

# Load the CSV into a data frame
diabetes_df = pd.read_csv("diabetes.csv")

# Use KNN imputation to fix 0s, except in columns where a 0 makes sense (pregnancies and outcome)
num_neighbors = 5
imputer = KNNImputer(n_neighbors=num_neighbors)
data_to_impute = diabetes_df.drop(["Pregnancies", "Outcome"], axis=1)

# Set "0" to null so it becomes imputed
data_to_impute = data_to_impute.replace({0:np.nan, 0.0:np.nan})

# Transform the data that needs imputation
imputed_diabetes_df = pd.DataFrame(imputer.fit_transform(data_to_impute), columns=data_to_impute.columns)

# Add back in Pregnancies and Outcome columns
imputed_diabetes_df["Pregnancies"] = diabetes_df["Pregnancies"]
imputed_diabetes_df["Outcome"] = diabetes_df["Outcome"]

# Save imputed dataset to a CSV for further use
imputed_diabetes_df.to_csv("cleaned_diabetes.csv", index=False)