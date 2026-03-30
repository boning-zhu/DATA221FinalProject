import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset into a data frame
cleaned_diabetes_data = pd.read_csv("Data/cleaned_diabetes.csv")

df = pd.read_csv("Data/cleaned_diabetes.csv")

# Split the dataset into input features (X) and target values (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

