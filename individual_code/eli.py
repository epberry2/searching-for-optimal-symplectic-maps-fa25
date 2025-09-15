import matplotlib.pyplot as plt
import numpy as np

# Adjustable parameters
num_objects = 1
object_type = 'circle'
num_comp = 10
degree = 5
num_points = 100

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
    
circle = Circle([3,1], 10)

x, y = circle.boundary_points(num_points)

fig, ax = plt.subplots()

ax.scatter(x, y, color='red')

ax.set_aspect('equal') 
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()


    