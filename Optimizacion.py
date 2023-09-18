import numpy as np
from scipy.optimize import minimize

# Definir la función f(X1, X2)
def f(X):
    x1, x2 = X
    return 10 - np.exp(-(x1**2 / 1 + 3 * x2**1.5))

# Calcular el gradiente de la función f(X1, X2)
def gradient(X):
    x1, x2 = X
    df_dx1 = (2 * x1 / 1) * np.exp(-(x1**2 / 1 + 3 * x2**1.5))
    df_dx2 = (3 * 1.5 * x2**0.5) * np.exp(-(x1**2 / 1 + 3 * x2**1.5))
    return np.array([df_dx1, df_dx2])

# Punto de inicio para la optimización
initial_guess = [0, 0]

# Encontrar el mínimo global de la función utilizando el método de BFGS
result = minimize(f, initial_guess, method='BFGS', jac=gradient)

# Imprimir el resultado
if result.success:
    print("Valor mínimo encontrado en X1 =", result.x[0])
    print("Valor mínimo encontrado en X2 =", result.x[1])
    print("Valor mínimo de la función =", result.fun)
else:
    print("La optimización no tuvo éxito. Mensaje de error:", result.message)
