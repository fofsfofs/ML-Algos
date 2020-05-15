import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hyp(theta, input):
    return theta.T.dot(input)


num_of_examples = 100
cols = ["x", "y"]
train = pd.read_csv("train.csv", names=cols)
np_vals = np.array(train.values)

x = np.delete(np_vals, 0, 0)[:, 0].astype(float)[:num_of_examples]
y = np.delete(np_vals, 0, 0)[:, 1].astype(float)[:num_of_examples]
theta = np.random.normal(size=2)
rate = 0.000001

print(f"Initial: {theta}")
# MATRIX MUST BE N BY D + 1 WHERE N = EXAMPLES AND D = FEATURES RN ITS 50x1


ones = np.full((num_of_examples), 1.0)
# print(ones.shape)
X = np.concatenate(
    (ones.reshape(num_of_examples, 1), x.reshape(num_of_examples, 1)), axis=1
)
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(f"Prediction: {theta}")


input = np.linspace(0, 100, 100)
output = theta[0] + theta[1] * input

plt.plot(input, output, "-r")
plt.scatter(x[:num_of_examples], y[:num_of_examples])
plt.show()
