# import libraries and modules
from utils.plot import plot_data, plot_decision_boundary
from model.nn import NeuralNetwork as nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
np.random.seed(0)

# reading data
data=pd.read_csv(r"Data\moons.csv")


# reshaping data
X=data[["x1","x2"]].to_numpy().reshape(-1,2)
Y=data.label.to_numpy().reshape(-1,1)

# 1. Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=72)

# 2. Initialize and train the model
model1 = nn(n_h=10, learning_rate=1.8, num_iterations=5000)
model1.fit(X_train, Y_train, print_cost=True)

# 3. Evaluate the model
acc = model1.score(X_test, Y_test)
print(f"Test Accuracy: {acc:.2%} \n ")
plot_data(X,Y)

# plot output
plot_decision_boundary(model1.predict, X, Y)


