import numpy as np

d = np.asarray([2, 1])
u0 = np.asarray([4, 4]) # initial guess

t_ini = 1
t_fin = 1e-9

def Function_Given(u, d):
    return np.log((u[0] - d[0]) * (u[1] - d[1]))


def constraints(u):
    return u[0] + 2 * u[1] - 12 <= 0 and 2 * u[0] + u[1] - 12<= 0 and u[0] - 2 >= 0 and u[1] - 1 >= 0

def grad_N(u, d):
    return np.array([1 / (u[0] - d[0]), 1 / (u[1] - d[1])])

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
    x1, x2, x3, x4 = u[0], u[1], d[0], d[1]
    dual_variables = np.linalg.inv(np.array([[1/(x1-x3), 0, 1, 0], [0, 1/(x2-x4), 0, 1], [2, 1, 0, -1], [1, 0, 0, 0]])).dot(np.array([0, 0, 0, 0]))
    return u, dual_variables



while t_ini / 2 > t_fin:
    t_ini /= 2
t_fin = t_ini / 2

u_star, dual_variables = newton_descent(u0, d, grad_N, hessian_N, constraints, t_ini, t_fin)

# (i)
print("Initial Value of t =",1)
print("Final Value of t =",t_fin)

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
print("lambda1 corresponding to u1 + 2*u2 <= 12: ", 1/((12-u_star[0]-(2*u_star[1])*t_fin)))
print("lambda2 corresponding to 2*u1 + u2 <= 12: ",1/((12-(2*u_star[0])*t_fin-(u_star[1]*t_fin))))


print("lambda1 corresponding to u1 + 2*u2 <= 12: ", dual_variables[0] )
print("lambda2 corresponding to 2*u1 + u2 <= 12: ", dual_variables[1])
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