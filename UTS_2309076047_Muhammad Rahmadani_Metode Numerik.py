import numpy as np
import matplotlib.pyplot as plt

# Constants for inductance, capacitance, and target frequency
L = 0.5  # Inductance (H)
C = 10e-6  # Capacitance (F)
target_f = 1000  # Target frequency (Hz)
tolerance = 1e-3  # Smaller tolerance for better precision

# Function to calculate resonance frequency based on resistance R
def f_R(R):
    term_sqrt = 1 / (L * C) - (R**2) / (4 * L**2)
    if term_sqrt <= 0:
        return None  # Return None if negative square root
    return (1 / (2 * np.pi)) * np.sqrt(term_sqrt)

# Derivative of f_R for Newton-Raphson method
def f_prime_R(R):
    term_sqrt = 1 / (L * C) - (R**2) / (4 * L**2)
    if term_sqrt <= 0:
        return None
    sqrt_term = np.sqrt(term_sqrt)
    return -R / (4 * np.pi * L**2 * sqrt_term)

# Newton-Raphson method
def newton_raphson_method(initial_guess, tolerance):
    R = initial_guess
    while True:
        f_val = f_R(R)
        if f_val is None:
            return None
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)
        if f_prime_value is None:
            return None
        new_R = R - f_value / f_prime_value
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Bisection method
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        mid = (a + b) / 2
        f_mid = f_R(mid) - target_f
        if f_mid is None:
            return None
        if abs(f_mid) < tolerance:
            return mid
        if (f_R(a) - target_f) * f_mid < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Initialize initial guesses
initial_guess = 50
interval_a, interval_b = 0, 100

# Execute methods
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_newton) if R_newton is not None else "Not found"
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) if R_bisection is not None else "Not found"

# Display results
print("Newton-Raphson Method:")
print(f"R: {R_newton} ohm, Resonance Frequency: {f_newton} Hz")
print("\nBisection Method:")
print(f"R: {R_bisection} ohm, Resonance Frequency: {f_bisection} Hz")

# Comparison plot
plt.figure(figsize=(10, 5))
plt.axhline(target_f, color="red", linestyle="--", label="Target Frequency 1000 Hz")

if R_newton is not None:
    plt.scatter(R_newton, f_newton, color="blue", label="Newton-Raphson")
    plt.text(R_newton, f_newton + 30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="blue")

if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color="green", label="Bisection")
    plt.text(R_bisection, f_bisection + 30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="green")

plt.xlabel("Resistance R (Ohm)")
plt.ylabel("Resonance Frequency f(R) (Hz)")
plt.title("Comparison of Newton-Raphson and Bisection Methods")
plt.legend()
plt.grid(True)
plt.show()

# Gauss and Gauss-Jordan elimination
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)
b = np.array([6, 2, 10], dtype=float)

def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        for j in range(i + 1, n):
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
    return x

def gauss_jordan(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]
    return Ab[:, -1]

# Solve using Gauss and Gauss-Jordan elimination
sol_gauss = gauss_elimination(A, b)
sol_gauss_jordan = gauss_jordan(A, b)

print("Solution using Gaussian Elimination:")
print(f"x1 = {sol_gauss[0]}, x2 = {sol_gauss[1]}, x3 = {sol_gauss[2]}")
print("\nSolution using Gauss-Jordan Elimination:")
print(f"x1 = {sol_gauss_jordan[0]}, x2 = {sol_gauss_jordan[1]}, x3 = {sol_gauss_jordan[2]}")
