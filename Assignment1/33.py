import numpy as np

# Define g1
def g1(x1, x2):
    return 2*x2 - x1

# Define g2
def g2(x1, x2):
    return 2*x1 - x2

# Define g3
def g3(x1, x2):
    return 1 - x1 - x2

# Define the Potential Function
def potential(x1,x2):
    g1_val = g1(x1, x2)
    g2_val = g2(x1, x2)
    g3_val = g3(x1, x2)
    gradient = 1/(g1_val*g2_val*g3_val) # As suggested in the Hint
    return gradient

#Calculating the gradient of the function by taking the partial derivative wrt to x1 and x2 from the functions g1,g2,g3
def gradient_Ψ(x1, x2):
    g1_x1 = -2
    g1_x2 = 2
    g2_x1 = 2
    g2_x2 = -2
    g3_x1 = -1
    g3_x2 = -1
    # The denominator is defined so because the value of log(g1(x1,x2))**-1 can be written as 1/g1(x1,x2) as mentioned in the Hint we can minimize f(x) instead of log(f(x))
    denominator = (1/g1(x1, x2)*g2(x1, x2)*g3(x1, x2))
    # Finding the gradients of x1 and x2 with the help of g1,g2,g3
    gradient_x1 = (g1_x1 / denominator + g2_x1 / denominator + g3_x1 / denominator)
    gradient_x2 = (g1_x2 / denominator + g2_x2 / denominator + g3_x2 / denominator)
    return gradient_x1, gradient_x2

#Calculating the Combination Descent(Minimum) of the Function using the Formula New_Value = Old_Value - (Learning_Rate * Slope)
def combination_descent(x1,x2):
    for i in range(iterations):
        gradientx1,gradientx2 = gradient_Ψ(x1, x2)
        x1 = x1 - learning_rate * gradientx1
        x2 = x2 - learning_rate * gradientx2
        if np.linalg.norm(gradient_Ψ(x1, x2)) <= tolerance:
            break
    return x1, x2

#Setting the Learning Rate to Lower Value
learning_rate = 0.01 

#Setting Random Value of the number of Iterations
iterations = 1000
tolerance = 1e-6

#Setting the values of x1 & x2 according to the questions
x1 = 0.25
x2 = 0.25

#Calculating the center of S
x1, x2 = combination_descent(x1,x2)
# Print the analytic center of S 
print(f"The analytic center is at ({x1:.2f}, {x2:.2f})")
