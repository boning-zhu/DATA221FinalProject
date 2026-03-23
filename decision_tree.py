import pandas as pd
import numpy as np

data = pd.read_csv("data/cleaned_diabetes.csv")
data.head()

data.info()
data.describe()