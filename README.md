# searching-for-optimal-symplectic-maps-fa25
This is the code for the fall 2025 IML project, Searching for Optimal Symplectic Maps.

The goal of the project is to embed regions in R^4 into the smallest possible ball. We achieve this by using an optimization algorithm to minimize the maximum distance of the region from the origin. 

To parameterize the space of symplectic maps, we approximate symplectic maps by the composition of Henon-like maps, 
(x,y) -> (y + n, -x + gradV(y)), where n is a vector in R^2, V : R^2 -> R is a polynomial in 2 variables.
Thus any symplectic map can be approximated by the composition of k Henon-like maps, where each Henon-like map is associated with a constant vector and polynomial of constant degree.

Therefore in the algorithm, we represent a Henon-like map as a d x d matrix and vector with 2 entries. Note that a d x d matrix represents a polynomial of degree 2d - 2, but not every term is realized. We flatten these terms to get an array with d^2 + 2 entries. Finally, we compose k of these maps, giving us an array of size k(d^2 + 2) to approximate any symplectic map.

Next, we want to find a symplectic map phi such that the function max||phi(x_i)|| is minimized, where x_1,...,x_n are points on the boundary of our region of interest. This max function is hard to work with, so we instead use a smooth approximation to the max function, the LogSumExp function. 
Now that everything is defined correctly, we use the Adam Optimizer from the machine learning library JAX to minimize this function, giving us our results.


Here is an overview of the structure of the code:

-Define the loss function to be optimized, also including regularization terms.
-Define evaluating one Henon-like map on some points.
-Define evaluating the composition of Henon-like maps on some points given an array of size k(d^2+2).
-Define our regions of interest: The ellipsoid E(1,a), the lagrangian tori L(1,a), the polydisc P(1,a).
-Define our training function with hyperparameters:
    -d = size of matrix for the Henon-like maps, representing a 2d - 2 polynomial
    -k = number of compositions
    -region = region of interest
    -n_boundary = number of points on the boundary of the region
    -n_iters = number of iterations 
    -lr = learning rate
    -tau = smoothing factor for loss function
-In the training function we use methods such as gradient clipping and learning rate back off for stability in the algorithm.
-After training it makes a report with the data at every iteration, saves the final parameters, saves animations of the training, and computes the symplectic error.

The hyperparameters can be adjusted in the main function at the bottom of the code.
Our best runs with the ellipsoid used d between 2-4 and k anywhere from 20-80. 
For accurate results, a large number of boundary points must be sampled, at least 10,000. 
For the lagrangian tori and polydisc, we were not able to make much improvements beyond the trivial bound.
