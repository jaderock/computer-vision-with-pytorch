import numpy as np
from numpy.linalg import inv  # for matrix inverse
import matplotlib.pyplot as plt

n = 11  # number of train data pairs (xi, yi) for nonlinear regression
scale = 0.7  # the amplitude of noise added to the 11 train data pairs (xi,yi)
f = lambda x: x**4 + x**2 - 1  # function for a smooth curve as a ground truth

x_true = np.linspace(-1.0, 1, 101)  # The ground truth curve's X
y_true = f(x_true)                  # The ground truth curve's Y=f(X)
x_train = np.linspace(-1, 1, n)     # Train data X
# Train data Y: noisy curve
y_train = (f(x_train) + np.random.uniform(-scale, scale, size=x_train.shape))

# A function for calculations of y_hat (mu) and weights
def y_hat(x, y, m):    # m: the order of a univariate polynomial
    ones = np.ones(len(x)) # ones=[1,1,...,1], len(ones)=n=11
    for i in range(1, m+1):
        ones = np.vstack((ones, x**i))
    X = ones.T
    W = inv(X.T@X)@(X.T@y)
    mu = X@W
    return mu, W

Y3hat, W3 = y_hat(x_train, y_train, 3)
Y9hat, W9 = y_hat(x_train, y_train, 9)


# Showing regression results
fig, ax = plt.subplots()
ax.plot(x_true, y_true, 'g-', label='y=x^4+x^2-1')
ax.plot(x_train, y_train, 'bp', label='train data')
ax.plot(x_train, Y3hat, 'r-', label='fitting, m=3')
ax.plot(x_train, Y9hat, 'k:', label='fitting, m=9')
ax.set(xlabel='x', ylabel='y')
ax.grid(color='g', linestyle=':')
ax.legend()
plt.show()

print(((Y3hat-y_train)**2).mean())
print(((Y9hat-y_train)**2).mean())
