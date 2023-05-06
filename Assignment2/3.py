import numpy as np

# Define the disagreement point
d = np.array([2, 1])

# Find the optimal solution using Newton's method
u0 = np.array([4, 4]) # initial guess
t_initial = 1
t_final = 1e-6

# Define the feasible set
def constrains(u):
    return u[0] + 2 * u[1] - 12 <= 0 and 2 * u[0] + u[1] - 12<= 0 and u[0] - 2 >= 0 and u[1] - 1 >= 0

# Define the objective function
def Function_Given(u, d):
    return np.log(u[0] - d[0]) + np.log(u[1] - d[1])

# Define the gradient of the objective function
def grad_N(u, d):
    return np.array([1 / (u[0] - d[0]), 1 / (u[1] - d[1])])

# Define the Hessian of the objective function
def hessian_N(u, d):
    return np.array([[-1 / ((u[0] - d[0]) ** 2), 0], [0, -1 / ((u[1] - d[1]) ** 2)]])

def newton_descent(u0, d, grad, hess, feasible, t_initial, t_final):
    u = u0
    t = t_initial
    while not feasible(u):
        hess_inv = np.linalg.inv(hess(u, d))
        while True:
            u_new = u - t * hess_inv.dot(grad(u, d))
            if feasible(u_new):
                u = u_new
                break
            else:
                t /= 2
        t = max(t / 2, t_final)
    
    # Calculate the optimal dual variables for the constraints using the KKT conditions
    x1, x2, x3, x4 = u[0], u[1], d[0], d[1]
    dual_variables = np.linalg.inv(np.array([[1/(x1-x3), 0, 1, 0], [0, 1/(x2-x4), 0, 1], [2, 1, 0, -1], [1, 0, 0, 0]])).dot(np.array([0, 0, 0, 0]))
    return u, dual_variables


# Calculate the final value of t
while t_initial / 2 > t_final:
    t_initial /= 2
t_final = t_initial / 2

u_star, dual_variables = newton_descent(u0, d, grad_N, hessian_N, constrains, t_initial, t_final)

# (i)
print("Initial Value of t =",1)
print("Final Value of t =",t_final)

# Calculate the welfare function value at the optimal solution
N_star = Function_Given(u_star, d)



# (ii)
print("Optimal solution:")
print("u1 =", u_star[0])
print("u2 =", u_star[1])

#(iii)
print("The Primal Optimal Solution is:", N_star)

#(iv)
print("Dual variables:")
print("lambda1 corresponding to u1 + 2*u2 <= 12: ", 1/((12-u_star[0]-(2*u_star[1])*t_final)))
print("lambda2 corresponding to 2*u1 + u2 <= 12: ",1/((12-(2*u_star[0])*t_final-(u_star[1]*t_final))))
print("lambda3 corresponding to u1 >= d1:        ", dual_variables[2])
print("lambda4 corresponding to u2 >= d2:        ", dual_variables[3])


#(v)
c1 = u_star[0] + 2 * u_star[1] - 12
c2 = 2 * u_star[0] + u_star[1] - 12
c3 = 2 - u_star[0]
c4 = 1 - u_star[1]

print("Inequality constraint function values at the optimum:")
print("Constraint value corresponding to u1 + 2*u2 <= 12: ", c1)
print("Constraint value corresponding to 2*u1 + u2 <= 12: ", c2)
print("Constraint value corresponding to u1 >= d1:       ", c3)
print("Constraint value corresponding to u2 >= d2:       ", c4)