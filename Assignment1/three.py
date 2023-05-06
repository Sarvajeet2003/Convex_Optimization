# import numpy as np

# #Setting the values of x1 & x2 according to the questions
# x1 = 0.25
# x2 = 0.25

# # Define g1
# def g1(x1,x2):
#     return 2*x2 - x1

# # Define g2
# def g2(x1,x2):
#     return 2*x1 - x2

# # Define g3
# def g3(x1,x2):
#     return 1 - x1 - x2

# # Define the Potential Function
# def Potential(x1,x2):
#     return 1/(g1(x1,x2)*g2(x1,x2)*g3(x1,x2))

# #Calculating the Combination Descent(Minimum) of the Function using the Formula New_Value = Old_Value + (Learning_Rate * Slope)
# def combination_descent(x1,x2):
#     for i in range(iterations):
#         x1 = x1 - (learning_rate * (-1 / g1(x1,x2)))
#         x2 = x2 - (learning_rate * (-1 / g2(x1,x2)))
#         if ((g1(x1, x2) >= 0) or (g2(x1, x2) >= 0) or (g3(x1, x2) >= 0)):
#             break;
#     return x1, x2

# #Setting the Learning Rate to Lower Value
# learning_rate = 0.001

# #Setting Random Value of the number of Iterations
# iterations = 1000

# #Calculating the center of S
# result_x1, result_x2 = combination_descent(x1, x2)

# # Print the analytic center of S 
# print(f"Analytic center of S ({result_x1},{result_x2})")



# import numpy as np

# def g1(x):
#     return 2 * x[1] - x[0]

# def g2(x):
#     return 2 * x[0] - x[1]

# def g3(x):
#     return 1 - x[0] - x[1]

# def Ψ(x):
#     return -np.log(g1(x)) - np.log(g2(x)) - np.log(g3(x))

# def gradient_Ψ(x):
#     g1_val = g1(x)
#     g2_val = g2(x)
#     g3_val = g3(x)
#     gradient = np.array([-1/g1_val + 1/g2_val, -1/g1_val - 1/g3_val])
#     return gradient

# def combination_descent(x0, a, max_iterations):
#     x = x0
#     for i in range(max_iterations):
#         gradient = gradient_Ψ(x)
#         x = x - a * gradient
#         if np.linalg.norm(gradient) < 1e-6:
#             break
#     return x

# x0 = np.array([0.25, 0.25])
# a = 0.1
# max_iterations = 1000

# analytic_center = combination_descent(x0, a, max_iterations)
# print("Analytic center:", analytic_center)



# import numpy as np

# def g1(x1, x2):
#     return (2 * x2) - x1

# def g2(x1, x2):
#     return (2 * x1) - x2

# def g3(x1, x2):
#     return 1 - x1 - x2

# def Ψ(x1, x2):
#     return -np.log(g1(x1, x2)) - np.log(g2(x1, x2)) - np.log(g3(x1, x2))

# # def gradient_Ψ(x1, x2):
    # g1_val = g1(x1, x2)
    # g2_val = g2(x1, x2)
    # g3_val = g3(x1, x2)
# #     gradient = 1/(g1_val*g2_val*g3_val)
# #     return gradient
# def gradient_Ψ(x1, x2):
#     g1_x1 = -2
#     g1_x2 = 2
#     g2_x1 = 2
#     g2_x2 = -2
#     g3_x1 = -1
#     g3_x2 = -1
#     denominator = min(g1(x1, x2), g2(x1, x2), g3(x1, x2))
#     gradient_x1 = -(g1_x1 / denominator + g2_x1 / denominator + g3_x1 / denominator)
#     gradient_x2 = -(g1_x2 / denominator + g2_x2 / denominator + g3_x2 / denominator)
#     return gradient_x1, gradient_x2


# def combination_descent(x1,x2, a, max_iterations):
#     for i in range(max_iterations):
#         gx1,gx2= gradient_Ψ(x1,x2)
#         x1 = x1 - (a * gx1)
#         x2 = x2 - (a * gx2)
#         # if ((g1(x1, x2) >= 0) or (g2(x1, x2) >= 0) or (g3(x1, x2) >= 0)):
#         #     break;
#     return x1,x2

# x1 = 0.25
# x2 = 0.25
# a = 0.001
# max_iterations = 1000

# analytic_center1,analytic_center2 = combination_descent(x1,x2, a, max_iterations)
# print("Analytic center:", analytic_center1,analytic_center2)






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
    return x1/3, x2/3

#Setting the Learning Rate to Lower Value
learning_rate = 0.1 

#Setting Random Value of the number of Iterations
iterations = 1000
tolerance = 1e-6

#Setting the values of x1 & x2 according to the questions
x1 = 0.25
x2 = 0.25

#Calculating the center of S
x1, x2 = combination_descent(x1,x2)
# Print the analytic center of S 
print(f"The analytic center is at ({x1/3:.6f}, {x2/3:.6f})")
