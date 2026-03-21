from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

diabetes_df = pd.read_csv("diabetes.csv")

num_neighbors = 5
imputer = KNNImputer(n_neighbors=num_neighbors)
df_to_impute = diabetes_df.drop(["Pregnancies", "Outcome"], axis=1)

df_to_impute = df_to_impute.replace({0:np.nan, 0.0:np.nan})

imputed_diabetes_df = pd.DataFrame(imputer.fit_transform(df_to_impute), columns=df_to_impute.columns)

imputed_diabetes_df["Pregnancies"] = diabetes_df["Pregnancies"]
imputed_diabetes_df["Outcome"] = diabetes_df["Outcome"]
print(imputed_diabetes_df)

imputed_diabetes_df.to_csv("cleaned_diabetes.csv", index=False)