# searching-for-optimal-symplectic-maps-fa25
This is the code for the fall 2025 IML project, Searching for Optimal Symplectic Maps.

The goal of this code is to find optimal symplectic maps in R^2. Here is the outline of what the code should do:

1. Define interesting regions. Since we already know what should happen in R^2, we should consider more interesting regions to map from.
For example, we should consider m number of circles in various locations in the plane, and see how they are mapped. 
We can also do this with rectangles. Whatever region we choose, we will call U.

2. Choose n random points on the boundary of our region, P_i for i = 1,..,n. For each iteration, we should change the points so that we do not only optimize the points we choose in the beginning.

3. Consider the Hamiltonians H = f(x) and H = g(y), where f and g are polynomials of 1 variable of fixed degree d. Then we can consider the flow of k compositions, phi_{AB} = A_1B_1 ... A_kB_k. 

4. Now we will look at phi_{AB}(P_i) for some i which is a vector in R^2. Let F_r = max||phi_{AB}(P_i)||, where r is our current iteration. The goal of this project is to minimize F, so we will use Adam Gradient Descent to do so. From There we get a new flow, 
phi_{A'B'} and we repeat the process.

5. Continue gradient descent until we reach a certain threshold.
