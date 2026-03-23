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

# train test split with random_state 42
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)