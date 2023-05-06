import numpy as np

#Defining the Fuction at F(x) = (x^2 − 3y^2)^2 + sin^2(x^2 + y^2)
def f(x,y):
    return pow((pow(x,2) - 3 * pow(y,2)),2) + pow((np.sin(pow(x,2) + pow(y,2))),2);

#Calculating the Gradient of the Function by Finding First order derivative wrt X,Y
def grad_f(x, y):
    df_dx = 4 * x * ( pow(x,2) - 3*pow(y,2) + np.sin(pow(x,2) + pow(y,2))*np.cos(pow(x,2)+pow(y,2)))
    df_dy = (-12*y*(pow(x,2)-3*pow(y,2))) + (4 * y * np.sin(pow(x,2)+pow(y,2))*np.cos(pow(x,2)+pow(y,2)))
    return df_dx, df_dy;

#Calculating the Combination Descent(Maximum) of the Function using the Formula New_Value = Old_Value - (Learning_Rate * Slope)
def combination_descent(f, grad_f):
    x = x0;
    y = y0;
    for i in range(iterations):
        dx, dy = grad_f(x, y)
        x = x - (learning_rate * dx)
        y = y - (learning_rate * dy)
    return x, y;

# Initilizing the values of X,Y to 1 as Given in the question
x0 = 1;
y0 = 1;

#Setting the Learning Rate to Lower Value
learning_rate = 0.001;

#Setting Random Value of the number of Iterations
iterations = 1000;

#Calculating the Y_Min and X_Min of the given function at X,Y = 1,1
x_min , y_min = combination_descent(f,grad_f);
print(f"Minimum point: ({x_min:.6f}, {y_min:.6f})");

#Calculating the Minimum Value of the Fuction at F(x) = (x^2 − 3y^2)^2 + sin^2(x^2 + y^2)
min = f(x_min,y_min);

#Printing the full Minimum Value 
print(f"Minimum value: {min}");

#Printing the Minimum Value upto 6 decimal places
print(f"Minimum value (Rounded off to 6 Decimal Places): {min:.6f}");