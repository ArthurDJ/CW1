import math
import numpy as np
import matplotlib.pyplot as plt

a = (1 + math.sqrt(5)) / 2
b = (1 - math.sqrt(5)) / 2

x = np.arange(-5, 5, (5 + 5) / 100)
x.dtype = complex
x.imag = 0
print(x)

y = (np.power(a, x) - np.power(b, x)) / np.sqrt(5)
print(y)

# plt.plot(x, y.real, label="real of y")
# plt.plot(x, y.imag, label="imag of y")
plt.plot(y.real, y.imag, label="x:real of fib, y:imag of fib")
plt.grid()
plt.legend()
plt.title("Fibonacci_interval")
plt.show()
