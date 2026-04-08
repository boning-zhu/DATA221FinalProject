import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the cleaned dataset into a data frame
df = pd.read_csv("Data/cleaned_diabetes.csv")

# Split the dataset into input features (X) and target values (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a range of k values from 1 to 10
k_values = range(1, 11)

# Store the accuracy for each k value
accuracy = []

# Train and test the KNN model for each k value
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)

    # Print the accuracy for each k
    print("k =", k, "Accuracy =", acc)

# Find the best k value
best_k = list(k_values)[accuracy.index(max(accuracy))]

# Train the final model using the best k value
best_model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
best_model.fit(X_train, y_train)
best_y_pred = best_model.predict(X_test)

# Calculate the final F1 score
final_f1 = f1_score(y_test, best_y_pred)

# Print the best k value, best accuracy, and F1 score
print("\nBest k:", best_k)
print("Best Accuracy:", max(accuracy))
print("F1 Score:", final_f1)

# Plot the accuracy for each k value
plt.plot(list(k_values), accuracy, marker="o")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()