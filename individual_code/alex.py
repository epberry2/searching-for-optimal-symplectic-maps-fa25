import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import csv



class Shape:
    def area(self):
        return 1.0  # default unit area

    def boundary_points(self, k = 1, seed = 0):
        raise NotImplementedError("boundary_points not implemented for base Shape class.")


class Keyhole(Shape):

    def __init__(self, position = [0,0], inner_radius = 1, outer_radius = 2, angle = np.pi/8):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle


    def boundary_points(self, k = 1, seed = 0):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1)
        rng3 = np.random.default_rng(seed + 2)
        rng4 = np.random.default_rng(seed + 3)
        theta1 = rng1.uniform(self.angle, 2 * np.pi - self.angle, k)
        theta2 = rng2.uniform(self.angle, 2 * np.pi - self.angle, k)
        t1 = rng3.uniform(0, 1, k)
        t2 = rng4.uniform(0, 1, k)
        
        x_in = self.inner_radius * np.cos(theta1) + self.position[0]
        y_in = self.inner_radius * np.sin(theta1) + self.position[1] 
        x_out = self.outer_radius * np.cos(theta2) + self.position[0]
        y_out = self.outer_radius * np.sin(theta2) + self.position[1]
        a1 = (self.inner_radius - t1 * (self.inner_radius - self.outer_radius)) * np.cos(self.angle) + self.position[0]
        b1 = (self.inner_radius - t1 * (self.inner_radius - self.outer_radius)) * np.sin(self.angle) + self.position[1]
        a2 = (self.outer_radius - t2 * (self.outer_radius - self.inner_radius)) * np.cos(2 * np.pi - self.angle) + self.position[0]
        b2 = (self.outer_radius - t2 * (self.outer_radius - self.inner_radius)) * np.sin(2 * np.pi - self.angle) + self.position[1]
        return x_in, y_in, x_out, y_out, a1, b1, a2, b2

    def area(self):
        return (self.outer_radius**2 - self.inner_radius**2)*(np.pi - self.angle/2)

# Create a Keyhole instance
keyhole = Keyhole(position=[0, 0], inner_radius=1, outer_radius=2, angle=np.pi/8)

# Generate boundary points
# Number of boundary points to sample
k = 500
x_in, y_in, x_out, y_out, a1, b1, a2, b2 = keyhole.boundary_points(k = k, seed=42)

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(x_in, y_in, color='blue', s=5, label='Inner Boundary')
plt.scatter(x_out, y_out, color='red', s=5, label='Outer Boundary')
plt.scatter(a1, b1, color='green', s=5)
plt.scatter(a2, b2, color='green', s=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Boundary Points of Keyhole Shape")
plt.legend()
plt.grid(True)
plt.show()
