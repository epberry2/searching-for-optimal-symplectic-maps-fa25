import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Adjustable parameters
num_objects = 1 # (does nothing yet)
object_type = 'circle'

num_compositions = 3 # number of compositions of A_i B_i
degree = 5 # degree of polynomial
num_points = 1000 # number of points on boundary of region

# randomly generated polynomial based on degree and nuimber of compositions
poly1 = np.random.uniform(-0.25, 0.25, 2 * num_compositions * (degree + 1))

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

# Class for symplectic map based on x -> x, y -> y + (df/dx)(x).
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

# Class for symplectic map based on x -> x + (dg/dy)(y), y -> y.
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
    
# Class for composition of symplectic maps
#
# This takes in an array of coefficients, an integer for degree, and an integer for number of compositions 
# The coefficients look like [a_10, a_11, ..., a_1d, b_10, b_11, ..., b_1d, a_20, ......, b_kd], a total of 2k(d+1) parameters
# Then the composition will look like phi_AB = A_1 B_1 A_2 B_2 ... A_k B_k
class Symplectic_Composition():

    def __init__(self, coeffs, degree, compositions):
        assert len(coeffs) == 2 * compositions * (degree + 1), "number of coefficients must be equal to 2k(d+1)"
        self.degree = degree
        self.comp = compositions
        self.coeffs = coeffs

        #initialize composition of maps
        a_list = []
        b_list = []
        for i in range(0, compositions):
            poly_a = np.array(coeffs[2 * i * (degree + 1) : 2 * i * (degree + 1) + degree + 1])
            poly_b = np.array(coeffs[2 * i * (degree + 1) + degree + 1 : 2 * (degree + 1) * (i + 1)])
            a_list.append(ASymplectic(poly_a))
            b_list.append(BSymplectic(poly_b))
        self.a_maps = a_list
        self.b_maps = b_list
    
    def compute_points(self, x, y):
        computed_x, computed_y = x, y
        for i in range(self.comp - 1, -1, -1):
            computed_x, computed_y = self.b_maps[i].compute_points(computed_x, computed_y)
            computed_x, computed_y = self.a_maps[i].compute_points(computed_x, computed_y)
        return computed_x, computed_y


def max_dist(x,y):
    return np.max(np.hypot(x,y))



# Test that the composition works correctly

# Choose random polynomials. Note for larger coefficients it gets weird.



# make the symplectic maps based off the polynomials
phi = Symplectic_Composition(poly1, degree, num_compositions)

# make a circle. Note for larger radius or position it gets weird.
circle = Circle([0,0], 0.5)

# Choose points on boundary and compute phi(points)
x, y = circle.boundary_points(num_points)
computed_x, computed_y = phi.compute_points(x, y)
disk_radius = max_dist(computed_x, computed_y)

# Plot regions
fig, ax = plt.subplots()

ax.scatter(x, y, color='red', s = 5)
ax.scatter(computed_x, computed_y, color='blue', s=5)

disk = patches.Circle((0, 0), radius=disk_radius, fill=False, edgecolor='purple', linewidth=2)
ax.add_patch(disk)

ax.set_aspect('equal') 
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()



    