import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hyp(theta, input):
    return theta.T.dot(input)


def grad_descend(theta, x, y, rate, iteration):
    theta_prime = np.copy(theta)
    bools = np.zeros(theta_prime.shape)

    for i in range(theta_prime.size):
        sum = 0
        for j in range(x.size):
            input = np.array([[1], [x[j]]])
            sum += (y[j] - hyp(theta_prime, input)) * x[j]
        theta_prime[i] += rate * sum
        if abs(theta[i] - theta_prime[i]) <= 0.0001:
            bools[i] = 1
    print(f"Iteration {iteration}: {theta_prime}")
    if np.alltrue(bools) == True:
        return theta_prime
    else:
        iteration += 1
        return grad_descend(theta_prime, x, y, rate, iteration)


num_of_examples = 50
cols = ["x", "y"]
train = pd.read_csv("train.csv", names=cols)
np_vals = np.array(train.values)

x = np.delete(np_vals, 0, 0)[:, 0].astype(float)[:num_of_examples]
y = np.delete(np_vals, 0, 0)[:, 1].astype(float)[:num_of_examples]
theta = np.random.normal(size=2)
rate = 0.000001

print(f"Initial: {theta}")
theta = grad_descend(theta, x, y, rate, 1)

input = np.linspace(0, 100, 100)
output = theta[0] + theta[1] * input

plt.plot(input, output, "-r")
plt.scatter(x[:num_of_examples], y[:num_of_examples])
plt.show()
