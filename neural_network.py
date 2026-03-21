import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# Load the cleaned dataset into a data frame
cleaned_diabetes_data = pd.read_csv("cleaned_diabetes.csv")

# Set seed so model always returns the same results
tf.random.set_seed(42)

# Set up neural network
neural_network_model = Sequential()

# Define layers
input_layer = InputLayer(input_shape=(8,)) # 8 input neurons for 8 features
neural_network_model.add(input_layer)
hidden_layer = Dense(3)
neural_network_model.add(hidden_layer)
output_layer=Dense(1, activation='sigmoid')
neural_network_model.add(output_layer)

# Compile the mode
neural_network_model.compile(loss='binary_crossentropy') # Using binary_crossentropy since this is a binary classification problem
