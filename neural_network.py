import pandas as pd # for reading CSV
import tensorflow as tf # for neural network
from tensorflow.keras.models import Sequential # for neural network

# Load the cleaned dataset into a data frame
cleaned_diabetes_data = pd.read_csv("cleaned_diabetes.csv")

# Set seed so model always returns the same results
tf.random.set_seed(42)

# Set up neural network
neural_network_model = Sequential()