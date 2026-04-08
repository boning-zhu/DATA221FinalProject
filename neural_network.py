import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split
import tensorflow.keras.metrics

# Load the cleaned dataset into a data frame
cleaned_diabetes_data = pd.read_csv("cleaned_diabetes.csv")

# Split data into training and testing
feature_matrix = cleaned_diabetes_data.drop(["Outcome"], axis=1)
target_class = cleaned_diabetes_data.loc[:, ["Outcome"]]
features_train, features_test, labels_train, labels_test = train_test_split(feature_matrix, target_class, test_size=.2, random_state=42)

# Set seed so model always returns the same results
tf.random.set_seed(42)
neural_network_model = Sequential()

# Define input layer
input_layer = InputLayer(input_shape=(8,)) # 8 input neurons for 8 features
neural_network_model.add(input_layer)

# Define hidden layer
hidden_layer = Dense(6)
neural_network_model.add(hidden_layer)
hidden_layer2 = Dense(4)
neural_network_model.add(hidden_layer2)

# Define output layer
output_layer=Dense(1, activation='sigmoid')
neural_network_model.add(output_layer)

# Compile the model
neural_network_model.compile(loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision', 'f1_score']) # Using binary_crossentropy since this is a binary classification problem

# Train the model
neural_network_model.fit(features_train, labels_train, epochs=40)

# Predict on testing data
class_probabilities = neural_network_model.predict(features_test)

# Evaluate the model
model_performance = neural_network_model.evaluate(features_test, labels_test)
print(model_performance)
print(f"Loss: {model_performance[0]}"
      f"\nAccuracy: {model_performance[1]}"
      f"\nRecall: {model_performance[2]}"
      f"\nPrecision: {model_performance[3]}"
      f"\nF1 Score: {model_performance[4]}")