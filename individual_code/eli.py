import matplotlib.pyplot as plt
import numpy as np

# Adjustable parameters
num_objects = 1
object_type = 'circle'
num_compositions = 10
degree = 5
num_points = 1000

class Shape:

    def __init__(self, position = [0,0]):
        self.position = position

    def boundary_points(self):
        pass

class Circle(Shape):

    def __init__(self, position = [0,0], radius = 1):
        super().__init__(position)
        self.radius = radius

    def boundary_points(self, k = 1):
        theta = np.random.uniform(0, 2 * np.pi, k)
        x = self.radius * np.cos(theta) + self.position[0]
        y = self.radius * np.sin(theta) + self.position[1]
        return x, y

# Class for symplectic map based on x -> x, y -> y + df/dx f(x).
class ASymplectic():

    def __init__(self, coeffs):
        self.degree = len(coeffs) - 1
        self.coeffs = coeffs

    def poly_derivative(self):
        return np.array([k * self.coeffs[k] for k in range(1, len(self.coeffs))])

    def compute_points(self, x, y):
        mapped_x = x
        mapped_y = y + np.polyval(self.poly_derivative()[::-1], x)
        return mapped_x, mapped_y

# Class for symplectic map based on x -> x + df/dy g(y), y -> y.
class BSymplectic():

    def __init__(self, coeffs):
        self.degree = len(coeffs) - 1
        self.coeffs = coeffs

    def poly_derivative(self):
        return np.array([k * self.coeffs[k] for k in range(1, len(self.coeffs))])

    def compute_points(self, x, y):
        mapped_x = x + np.polyval(self.poly_derivative()[::-1], y)
        mapped_y = y
        return mapped_x, mapped_y



# Test that AB maps correctly

# Choose random polynomials. Note for larger coefficients it gets weird.
poly1 = np.random.uniform(-0.5, 0.5, 6)
poly2 = np.random.uniform(-0.5, 0.5, 6)
print(poly1)
print(poly2)

# make the symplectic maps based off the polynomials
A = ASymplectic(poly1)
B = BSymplectic(poly2)

# make a circle. Note for larger radius or position it gets weird.
circle = Circle([0,0], 0.9)

# Choose points on boundary and computed AB(p_i)
x, y = circle.boundary_points(num_points)
B_x, B_y = B.compute_points(x, y)
computed_x, computed_y = A.compute_points(B_x, B_y)

# Plot regions
fig, ax = plt.subplots()

ax.scatter(x, y, color='red')
ax.scatter(computed_x, computed_y, color='blue')

ax.set_aspect('equal') 
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# function that defines symplectic map in terms of coefficents

# function that returns the max distance of the mapped points


    