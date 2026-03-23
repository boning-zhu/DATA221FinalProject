import pandas as pd
import numpy as np

data = pd.read_csv("Data/cleaned_diabetes.csv")
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

#import decision tree model
from sklearn.tree import DecisionTreeClassifier
#initialize decision tree model
model = DecisionTreeClassifier(random_state=42)
#train decisioin tree model
model.fit(X_train, y_train)
#make predictions
y_pred = model.predict(X_test)

#add evaluation metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#add confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#add feature importance plot
importance = model.feature_importances_

plt.barh(X.columns, importance)
plt.title("Feature Importance")
plt.show()

#optimize decision tree parameters
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    random_state=42
)

# optimize decision tree parameters
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    random_state=42
)

# retrain
model.fit(X_train, y_train)

# predict again
y_pred = model.predict(X_test)

# evaluation
from sklearn.metrics import classification_report, accuracy_score

print("Optimized Model Results:")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)