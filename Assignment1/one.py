import numpy as np
import matplotlib.pyplot as plt

# Defining the Fuction F(x) which is given in the question as f(x) = x^2 + log x
def f(x):
    return x**2 + np.log(x)

# Calculate the Linear Taylor polynomial
def L(x):
    return 1 + 2 * (x - 1)

# Calculate the Linear Taylor polynomial
def Q(x):
    return 1 + 2 * (x - 1) + (x - 1)**2

# Setting the Value of x as given in the Question
x = np.linspace(0, 2, 1000)

# Plotting the first graph i.e. (a)
plt.plot(x, f(x), 'r', label='f(x) = x^2 + log(x)')
plt.plot(x, L(x), 'g', label='L(x) = 1 + 2(x - 1)')
plt.plot(x, Q(x), 'b', label='Q(x) = 1 + 2(x - 1) + (x - 1)^2')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of f(x), L(x), and Q(x)')
plt.grid()
plt.show()


# Calculate the Function eQ(x)
def eQ(x, a):
    return f(x) - Q(x);

# Calculate the Function eL(x)
def eL(x, a):
    return f(x) - L(x);

# Plotting the Second graph i.e. (b)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, eL(x, 1)/(x-1), label='eL(x)/(x-1)')
plt.plot(x, eQ(x, 1)/pow((x-1),2), label='eQ(x)/(x-1)^2')
plt.legend()
plt.title(" Graphs of Errors ")
plt.show()