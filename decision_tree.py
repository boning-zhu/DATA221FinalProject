import pandas as pd
import numpy as np

data = pd.read_csv("data/cleaned_diabetes.csv")
data.head()

#explore dataset structure
data.info()
data.describe()

#splite features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]