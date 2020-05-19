# Linear Regression using normal equations which minimizes the cost function using matrix derivatives
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hyp(theta, input):
    # The hypothesis is theta transposed multiplied by the input(column vector)
    return theta.T.dot(input)


# Data processing
num_of_examples = 50
cols = ["x", "y"]
train = pd.read_csv("train.csv", names=cols)
# np_vals are all the values in the file
np_vals = np.array(train.values)

# Delete the labels ("x" and "y"), convert values to floats, and trim inputs and outputs to only hold specifiednumber of examples
x = np.delete(np_vals, 0, 0)[:, 0].astype(float)[:num_of_examples]
y = np.delete(np_vals, 0, 0)[:, 1].astype(float)[:num_of_examples]

# Randomize theta intially
theta = np.random.normal(size=2)

print(f"Initial: {theta}")


ones = np.full((num_of_examples), 1.0)
# The X matrix is an n x m matrix (n = examples, m = features) and all the input vectors are transposed
X = np.concatenate(
    (ones.reshape(num_of_examples, 1), x.reshape(num_of_examples, 1)), axis=1
)

# The magic equation after forming a bunch of matrix calculus
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(f"Prediction: {theta}")

# Plotting the line of best fit and the corresponding data
input = np.linspace(0, 100, 100)
output = theta[0] + theta[1] * input

plt.plot(input, output, "-r")
plt.scatter(x[:num_of_examples], y[:num_of_examples])
plt.show()
