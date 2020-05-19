# Linear Regression using gradient descent and using least squares as the cost function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hyp(theta, input):
    # The hypothesis is theta transposed multiplied by the input(column vector)
    return theta.T.dot(input)


def grad_descend(theta, x, y, rate, iteration):
    # Recursive gradient descent
    theta_prime = np.copy(theta)
    bools = np.zeros(theta_prime.shape)

    for i in range(theta_prime.size):
        sum = 0
        for j in range(x.size):
            # Input matrix where x0 = 1 by convention
            input = np.array([[1], [x[j]]])
            # Sum of least squares
            sum += (y[j] - hyp(theta_prime, input)) * x[j]
        theta_prime[i] += rate * sum
        # Change between previous theta index and current theta index is very small set corresponding boolean to true
        if abs(theta[i] - theta_prime[i]) <= 0.0001:
            bools[i] = 1
    print(f"Iteration {iteration}: {theta_prime}")

    # If the change the between all elements in the theta matrix are small, return theta else perform gradient descent again
    if np.alltrue(bools) == True:
        return theta_prime
    else:
        iteration += 1
        return grad_descend(theta_prime, x, y, rate, iteration)


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
# ↑ Number of examples use ↓ learning rate
rate = 0.000001

print(f"Initial: {theta}")

try:
    theta = grad_descend(theta, x, y, rate, 1)
except RecursionError as error:
    print("Could not converge!")

# Plotting the line of best fit and the corresponding data
input = np.linspace(0, 100, 100)
output = theta[0] + theta[1] * input

plt.plot(input, output, "-r")
plt.scatter(x[:num_of_examples], y[:num_of_examples])
plt.show()
