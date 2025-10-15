#!/usr/bin/env python3
# Boundary-only: learn a 2D symplectic map (composition of shears) that
# minimizes the maximum radius of the mapped boundary.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import csv
import time

class Shape4D:
    def volume(self):
        return 1.0  # default unit volume

    def boundary_points(self, k = 1, seed = 0):
        raise NotImplementedError("boundary_points not implemented for base Shape4D class.")
    
class Ellipsoid4D(Shape4D):

    def __init__(self, position = [0,0,0,0], radii = [1,1,1,1]):
        self.position = position
        self.radii = radii

    def boundary_points(self, k = 1, seed = 0):
        rng = np.random.default_rng(seed)
        u = rng.normal(0, 1, (4, k))
        norm = np.linalg.norm(u, axis=0)
        x = self.radii[0] * u[0,:] / norm + self.position[0]
        y = self.radii[1] * u[1,:] / norm + self.position[1]
        z = self.radii[2] * u[2,:] / norm + self.position[2]
        w = self.radii[3] * u[3,:] / norm + self.position[3]
        return x, y, z, w
    
    def volume(self):
        return (4/3)*np.pi**2 * np.prod(self.radii)
    
class Shape:
    def area(self):
        return 1.0  # default unit area

    def boundary_points(self, k = 1, seed = 0):
        raise NotImplementedError("boundary_points not implemented for base Shape class.")

class Circle(Shape):

    def __init__(self, position = [0,0], radius = 1):
        self.position = position
        self.radius = radius

    def boundary_points(self, k = 1, seed = 0):
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, k)
        x = self.radius * np.cos(theta) + self.position[0]
        y = self.radius * np.sin(theta) + self.position[1]
        return x, y
    
    def area(self):
        return np.pi * self.radius**2

class MultipleCircles(Shape):

    def __init__(self, positions, radii):
        self.positions = positions
        self.radii = radii

    def boundary_points(self, k=1, seed=0):
        rng = np.random.default_rng(seed)
  
        xs = []
        ys = []
        for pos, radius in zip(self.positions, self.radii):
            theta = rng.uniform(0, 2 * np.pi, k // len(self.positions))
            x = radius * np.cos(theta) + pos[0]
            y = radius * np.sin(theta) + pos[1]
            xs = np.concatenate([xs, x])
            ys = np.concatenate([ys, y])
        return xs, ys
    
    def area(self):
        return np.pi * sum(r**2 for r in self.radii)

class Rectangle(Shape):

    def __init__(self, position = [0,0], verthalfwidth = 0.5, horihalfwidth = 0.25):
        self.position = position
        self.verthalfwidth = verthalfwidth
        self.horihalfwidth = horihalfwidth

    def area(self):
        return 4 * self.verthalfwidth * self.horihalfwidth

    def boundary_points(self, n, seed=0):
        x, y = square_boundary_points(n, self.verthalfwidth, self.horihalfwidth, seed=seed)
        x += self.position[0]
        y += self.position[1]
        return x, y
    
class MultipleShapes(Shape):

    def __init__(self, shapes):
        self.shapes = shapes

    def boundary_points(self, k=1, seed=0):
        xs = []
        ys = []
        for shape in self.shapes:
            x, y = shape.boundary_points(k // len(self.shapes), seed=seed)
            xs = np.concatenate([xs, x])
            ys = np.concatenate([ys, y])
        return xs, ys
    
    def area(self):
        return sum(shape.area() for shape in self.shapes)

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
        theta1 = rng1.uniform(self.angle, 2 * np.pi - self.angle, k // 3)
        theta2 = rng2.uniform(self.angle, 2 * np.pi - self.angle, k // 3)
        t1 = rng3.uniform(0, 1, k // 6)
        t2 = rng4.uniform(0, 1, k // 6)
        
        x_in = self.inner_radius * np.cos(theta1) + self.position[0]
        y_in = self.inner_radius * np.sin(theta1) + self.position[1] 
        x_out = self.outer_radius * np.cos(theta2) + self.position[0]
        y_out = self.outer_radius * np.sin(theta2) + self.position[1]
        a1 = (self.inner_radius - t1 * (self.inner_radius - self.outer_radius)) * np.cos(self.angle) + self.position[0]
        b1 = (self.inner_radius - t1 * (self.inner_radius - self.outer_radius)) * np.sin(self.angle) + self.position[1]
        a2 = (self.outer_radius - t2 * (self.outer_radius - self.inner_radius)) * np.cos(2 * np.pi - self.angle) + self.position[0]
        b2 = (self.outer_radius - t2 * (self.outer_radius - self.inner_radius)) * np.sin(2 * np.pi - self.angle) + self.position[1]
        xs = np.concatenate([x_in, x_out, a1, a2])
        ys = np.concatenate([y_in, y_out, b1, b2])
        return xs, ys

    def area(self):
        return (self.outer_radius**2 - self.inner_radius**2)*(np.pi - self.angle)

# ---------------------------
# Boundary sampling (square)
# ---------------------------
def square_boundary_points(n, verthalfwidth=0.5, horihalfwidth = 0.25, seed=0):
    rng = np.random.default_rng(seed)
    n_side = [n // 4] * 4
    for i in range(n % 4):
        n_side[i] += 1
    xs, ys = [], []

    # Bottom: y=-h
    x = rng.uniform(-horihalfwidth, horihalfwidth, n_side[0])
    y = np.full_like(x, -verthalfwidth); xs.append(x); ys.append(y)
    # Right: x=+h
    y = rng.uniform(-verthalfwidth, verthalfwidth, n_side[1])
    x = np.full_like(y, +horihalfwidth); xs.append(x); ys.append(y)
    # Top: y=+h
    x = rng.uniform(-horihalfwidth, horihalfwidth, n_side[2])
    y = np.full_like(x, +verthalfwidth); xs.append(x); ys.append(y)
    # Left: x=-h
    y = rng.uniform(-verthalfwidth, verthalfwidth, n_side[3])
    x = np.full_like(y, -horihalfwidth); xs.append(x); ys.append(y)

    X = np.concatenate(xs); Y = np.concatenate(ys)
    return X, Y


# ---------------------------
# Exact symplectic shear maps
# ---------------------------
class ASymplecticR4:
    # A_f: (x1, x2, y1, y2) -> (x1, x2, y1 + df/dx1(x1,x2), y2 + df/dx2(x1,x2))
    # f(x1, x2) is a bivariate polynomial
    # coeffs: 2d array of size d+1 by d+1, coeffs[i,j] is coeff of x1^i * x2^j
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        d1, d2 = self.coeffs.shape
        assert d1 == d2, "Coefficient array must be square."
        da1 = np.array([[i * self.coeffs[i,j] for j in range(d2+1)] for i in range(1, d1+1)], dtype=float) # df/dx1
        da2 = np.array([[j * self.coeffs[i,j] for j in range(1, d2+1)] for i in range(d1+1)], dtype=float) # df/dx2
        self.dcoeffs1_desc = da1[::-1, ::-1]  # for np.polyval2d
        self.dcoeffs2_desc = da2[::-1, ::-1]
    def __call__(self, x1, x2, y1, y2):
        return x1, x2, y1 + np.polyval2d(x1, x2, self.dcoeffs1_desc), y2 + np.polyval2d(x1, x2, self.dcoeffs2_desc)
    
class BSymplecticR4:
    # B_g: (x1, x2, y1, y2) -> (x1 + dg/dy1(y1,y2), x2 + dg/dy2(y1,y2), y1, y2)
    # g(y1, y2) is a bivariate polynomial
    # coeffs: 2d array of size d+1 by d+1, coeffs[i,j] is coeff of y1^i * y2^j
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        d1, d2 = self.coeffs.shape
        assert d1 == d2, "Coefficient array must be square."
        db1 = np.array([[i * self.coeffs[i,j] for j in range(d2+1)] for i in range(1, d1+1)], dtype=float) # dg/dy1
        db2 = np.array([[j * self.coeffs[i,j] for j in range(1, d2+1)] for i in range(d1+1)], dtype=float) # dg/dy2
        self.dcoeffs1_desc = db1[::-1, ::-1]  # for np.polyval2d
        self.dcoeffs2_desc = db2[::-1, ::-1]
    def __call__(self, x1, x2, y1, y2):
        return x1 + np.polyval2d(y1, y2, self.dcoeffs1_desc), x2 + np.polyval2d(y1, y2, self.dcoeffs2_desc), y1, y2
    
class SymplecticCompositionR4:
    """
    Phi = A1 ∘ B1 ∘ ... ∘ Ak ∘ Bk.
    Apply to coords as: (x1,x2,y1,y2) -> Bk -> Ak -> ... -> B1 -> A1
    params: flat array length 2*k*(degree+1)*(degree+1)
    """
    def __init__(self, params, degree, k):
        self.degree = degree; self.k = k
        self.set_params(params)

    def set_params(self, params):
        params = np.asarray(params, dtype=float)
        D = self.degree + 1
        assert params.size == 2*self.k*D*D, "Parameter length mismatch."
        self._params = params.copy()
        self.A, self.B = [], []
        for i in range(self.k):
            a = params[2*i*D*D : 2*i*D*D + D*D].reshape((D,D))
            b = params[2*i*D*D + D*D : 2*(i+1)*D*D].reshape((D,D))
            self.A.append(ASymplecticR4(a))
            self.B.append(BSymplecticR4(b))

    def params(self): return self._params.copy()

    def forward(self, x1, x2, y1, y2):
        for i in range(self.k-1, -1, -1):
            x1, x2, y1, y2 = self.B[i](x1, x2, y1, y2)
            x1, x2, y1, y2 = self.A[i](x1, x2, y1, y2)
        return x1, x2, y1, y2

class ASymplectic:
    # A_f: (x,y) -> (x, y + f'(x))
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        d = len(self.coeffs) - 1
        dasc = np.array([k * self.coeffs[k] for k in range(1, d + 1)], dtype=float)
        self.dcoeffs_desc = dasc[::-1]  # for np.polyval
    def __call__(self, x, y):
        return x, y + np.polyval(self.dcoeffs_desc, x)

class BSymplectic:
    # B_g: (x,y) -> (x + g'(y), y)
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        d = len(self.coeffs) - 1
        dasc = np.array([k * self.coeffs[k] for k in range(1, d + 1)], dtype=float)
        self.dcoeffs_desc = dasc[::-1]
    def __call__(self, x, y):
        return x + np.polyval(self.dcoeffs_desc, y), y

class SymplecticComposition:
    """
    Phi = A1 ∘ B1 ∘ ... ∘ Ak ∘ Bk.
    Apply to coords as: (x,y) -> Bk -> Ak -> ... -> B1 -> A1
    params: flat array length 2*k*(degree+1)
    """
    def __init__(self, params, degree, k):
        self.degree = degree; self.k = k
        self.set_params(params)

    def set_params(self, params):
        params = np.asarray(params, dtype=float)
        D = self.degree + 1
        assert params.size == 2*self.k*D, "Parameter length mismatch."
        self._params = params.copy()
        self.A, self.B = [], []
        for i in range(self.k):
            a = params[2*i*D : 2*i*D + D]
            b = params[2*i*D + D : 2*(i+1)*D]
            self.A.append(ASymplectic(a))
            self.B.append(BSymplectic(b))

    def params(self): return self._params.copy()

    def forward(self, x, y):
        for i in range(self.k-1, -1, -1):
            x, y = self.B[i](x, y)
            x, y = self.A[i](x, y)
        return x, y

# ---------------------------
# Loss: true hard max on boundary
# ---------------------------
def loss_max_radius_boundary(Phi, xB, yB, w_center=1e-3, w_reg=1e-6):
    xb, yb = Phi.forward(xB, yB)
    r = np.hypot(xb, yb)
    R = float(r.max())  # true discrete max radius on boundary

    # gentle helpers for stability
    cx = float(xb.mean()); cy = float(yb.mean())
    reg = float(np.sum(Phi.params()**2))

    L = R + w_center*(cx*cx + cy*cy) + w_reg*reg
    aux = dict(R=R, cx=cx, cy=cy, rmean=float(r.mean()), rvar=float(r.var()))
    return L, aux, (xb, yb)

# ---------------------------
# Adam + finite-difference grads
# ---------------------------
class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=np.zeros_like(params); self.v=np.zeros_like(params); self.t=0
    def step(self, params, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*(grad*grad)
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return params - self.lr * mhat / (np.sqrt(vhat) + self.eps)

def finite_diff_grad(f, theta, eps=1e-4):
    g = np.zeros_like(theta, dtype=float)
    f0 = f(theta)
    for i in range(theta.size):
        t = theta.copy(); t[i] += eps
        g[i] = (f(t) - f0) / eps
    return g

def analytic_grad(Phi, xB, yB, w_center=1e-3, w_reg=5e-7):
    D = Phi.degree + 1
    theta = Phi.params()
    k = Phi.k

    preB_x  = [None]*k 
    preB_y  = [None]*k
    postB_x = [None]*k 
    postB_y = [None]*k
    postA_x = [None]*k 
    postA_y = [None]*k

    x, y = xB, yB
    for i in range(k-1, -1, -1):
        preB_x[i], preB_y[i] = x, y
        x, y = Phi.B[i](x, y)
        postB_x[i], postB_y[i] = x, y
        x, y = Phi.A[i](x, y)
        postA_x[i], postA_y[i] = x, y

    x_final, y_final = x, y

    r = np.hypot(x_final, y_final)
    R = r.max()
    mask = (r >= R - 0.0)

    gx = np.zeros_like(x_final)
    gy = np.zeros_like(y_final)
    gx[mask] = x_final[mask] / (r[mask] + 1e-12)
    gy[mask] = y_final[mask] / (r[mask] + 1e-12)

    gx += 2.0 * w_center * (x_final.mean()) / x_final.size
    gy += 2.0 * w_center * (y_final.mean()) / y_final.size
    grad = np.zeros_like(theta)
    for i in range(0, k):
        x_in_A = postB_x[i]
        a = Phi.A[i].coeffs  

        if D >= 3:
            d2_a = np.array([m*(m-1)*a[m] for m in range(2, D)], dtype=float)
            f2 = np.polyval(d2_a[::-1], x_in_A)
        else:
            f2 = 0.0
        base = 2*i*D
        for m in range(1, D):
            grad[base + m] += np.sum(gy * (x_in_A**(m-1)) * m)
        gx = gx + gy * f2
        y_in_B = preB_y[i]
        b = Phi.B[i].coeffs
        if D >= 3:
            d2_b = np.array([m*(m-1)*b[m] for m in range(2, D)], dtype=float)
            g2 = np.polyval(d2_b[::-1], y_in_B)
        else:
            g2 = 0.0
        for m in range(1, D):
            grad[base + D + m] += np.sum(gx * (y_in_B**(m-1)) * m)
        gy = gy + gx * g2
    grad += 2.0 * w_reg * theta
    return grad

# ---------------------------
# Training (boundary only)
# ---------------------------
def train_min_radius_boundary_2d(
    degree=5, k=3,
    n_boundary=3000, region=Shape(),
    n_iters=300, lr=2e-2, seed=7,
    polynomial_bound=0.005,
    w_center=1e-3, w_reg=5e-7, report_every=25,
    animate=False
):
    rng = np.random.default_rng(seed)
    xB, yB = region.boundary_points(n_boundary, seed=seed)

    area = region.area()
    r_eq = np.sqrt(area / np.pi)

    D = degree + 1
    num_params = 2*k*D
    theta0 = rng.uniform(-polynomial_bound, polynomial_bound, num_params)
    Phi = SymplecticComposition(theta0, degree, k)
    opt = Adam(Phi.params(), lr=lr)

    def fobj(theta):
        Phi.set_params(theta)
        L, _, _ = loss_max_radius_boundary(Phi, xB, yB, w_center=w_center, w_reg=w_reg)
        return L

    history = []
    best = (np.inf, Phi.params())
    bestiter = 0

    # For animation
    frames = []

    for it in range(1, n_iters+1):
        xB, yB = region.boundary_points(n_boundary, seed=seed + it)
        grad = analytic_grad(Phi, xB, yB, w_center=w_center, w_reg=w_reg)

        # stop if gradient contains NaN/Inf
        if not np.all(np.isfinite(grad)):
            print(f"[{it:4d}] Non-finite gradient encountered; stopping.")
            break

        new_params = opt.step(Phi.params(), grad)

        # stop if optimizer produced NaN/Inf parameters
        if not np.all(np.isfinite(new_params)):
            print(f"[{it:4d}] Non-finite parameters produced by optimizer; stopping.")
            break

        Phi.set_params(new_params)

        L, aux, (xb, yb) = loss_max_radius_boundary(Phi, xB, yB, w_center=w_center, w_reg=w_reg)

        # check for non-finite loss 
        if not np.isfinite(L):
            print(f"[{it:4d}] Non-finite loss encountered (L={L}); stopping.")
            break

        history.append(dict(it=it, loss=float(L), R=aux["R"], rmean=aux["rmean"],
                            rvar=aux["rvar"], cx=aux["cx"], cy=aux["cy"]))

        if L < best[0]:
            best = (float(L), Phi.params())
            bestiter = it

        if it % report_every == 0 or it == 1 or it == n_iters:
            print(f"[{it:4d}] L={L:.6f}  R={aux['R']:.6f}  r_eq={r_eq:.6f}  "
                  f"rmean={aux['rmean']:.6f}  var={aux['rvar']:.3e}  "
                  f"cent=({aux['cx']:.2e},{aux['cy']:.2e})")
            if animate:
                frames.append((xb.copy(), yb.copy()))

    Phi.set_params(best[1])
    xb, yb = Phi.forward(xB, yB)
    return Phi, (xB, yB), (xb, yb), r_eq, history, bestiter, frames if animate else None

# ---------------------------
# Demo
# ---------------------------

OUT_DIR = "output_r_vs_d"  # folder already in repo; safe to write here
os.makedirs(OUT_DIR, exist_ok=True)

def run_vary_degree(
    degree_values,
    k=10,
    n_boundary=3000,
    n_iters=300,
    seed=1,
    polynomial_bound=0.005,
    lr=2e-3,
    w_center=1e-3,
    w_reg=5e-7,
    report_every=1000,
    animate=False,
    region=None,
):
    if region is None:
        region = Keyhole(position=[0,0], inner_radius=0.25, outer_radius=0.75, angle=np.pi/8)

    results = []
    for d in degree_values:
        print(f"\n=== Running d={d} ===")
        t0 = time.time()
        Phi, (xB0, yB0), (xB1, yB1), r_eq, history, bestiter, frames = train_min_radius_boundary_2d(
            degree=d,
            k=k,
            n_boundary=n_boundary,
            region=region,
            n_iters=n_iters,
            lr=lr,
            seed=seed,
            polynomial_bound=polynomial_bound,
            w_center=w_center,
            w_reg=w_reg,
            report_every=report_every,
            animate=animate,
        )
        t1 = time.time()
        r = np.hypot(xB1, yB1)
        R = float(r.max())
        print(f"d={d} finished in {t1-t0:.1f}s  final true max radius R={R:.6f}")
        results.append(dict(d=int(d), R=float(R), r_eq=float(r_eq), time_s=(t1-t0)))

        # save a small scatter for each d (optional)
        fig, ax = plt.subplots(figsize=(4,4))
        ax.scatter(xB1, yB1, s=1)
        circ = plt.Circle((0,0), R, fill=False, linewidth=1.2, color='r')
        ax.add_patch(circ)
        ax.set_aspect('equal')
        ax.set_title(f'd={d}  R={R:.4f}')
        fname = os.path.join(OUT_DIR, f'map_d_{d}.png')
        plt.tight_layout(); plt.savefig(fname); plt.close(fig)

    # write csv
    csv_path = os.path.join(OUT_DIR, 'vary_d_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['d','R','r_eq','time_s'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # plot R vs d
    ds = [r['d'] for r in results]
    Rs = [r['R'] for r in results]
    plt.figure(figsize=(6,4))
    plt.plot(ds, Rs, marker='o')
    for i, t in enumerate([r['time_s'] for r in results]):
        plt.text(ds[i], Rs[i], f"{t:.1f}s", fontsize=8, ha='center', va='bottom')
    plt.xlabel('d (degree of polynomial)')
    plt.ylabel('Final true max radius R')
    plt.title('Final max radius vs d')
    plt.grid(True)
    out_png = os.path.join(OUT_DIR, 'vary_d_plot.png')
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"Results written to {csv_path} and {out_png}")

    return results

if __name__ == '__main__':
    # small default sweep (adjust as you like)
    d_values = [1, 2, 5, 6, 8, 10, 12]
    res = run_vary_degree(d_values, k=5, n_boundary=3000, n_iters=30000, lr=2e-3, seed=10)
    print('\nSummary:')
    for r in res:
        print(f"d={r['d']}: R={r['R']:.6f}  r_eq={r['r_eq']:.6f}  time={r['time_s']:.1f}s")

'''
if __name__ == "__main__":
    Phi, (xB0, yB0), (xB1, yB1), r_eq, hist, bestiter, frames = train_min_radius_boundary_2d(
        degree=5, k=10,
        n_boundary=5000, 
        region=Keyhole(position=[0, 0], inner_radius=0.25, outer_radius=0.75, angle=np.pi/8),
        polynomial_bound=0.005,
        n_iters=500, lr=2e-3, seed=10,
        w_center=1e-3, w_reg=5e-7, report_every=25,
        animate=True
    )

    # final stats
    r = np.hypot(xB1, yB1)
    R = float(r.max())
    print("\nFinal summary:")
    print(f"  Equal-area lower bound r_eq = {r_eq:.6f}")
    print(f"  True max radius (boundary)  = {R:.6f}")
    print(f"  Best Iteration               = {bestiter}")
    print(f"  Mean boundary radius        = {float(r.mean()):.6f}")

    # Print polynomials of the best map
    print("\nBest map polynomials (coefficients for A_i and B_i, lowest->highest degree):")
    for i in range(Phi.k):
        a_coeffs = Phi.A[i].coeffs
        b_coeffs = Phi.B[i].coeffs
        print(f"  A{i+1}: {a_coeffs.tolist()}")
        print(f"  B{i+1}: {b_coeffs.tolist()}")

    # write history to CSV
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "history.csv")
    fieldnames = ["it", "loss", "R", "rmean", "rvar", "cx", "cy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in hist:
            # write only the selected fields (safe if keys missing)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"History written to {csv_path}")

    # Animation
    if frames is not None:
        fig, ax = plt.subplots()
        scat = ax.scatter([], [], s=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title("Boundary evolution")

        def update(frame):
            x, y = frame
            scat.set_offsets(np.column_stack([x, y]))
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
        ani.save("test1.mp4", writer="ffmpeg") # use ffmpeg to save as mp4, use pillow to save as gif

        plt.show()

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(xB0, yB0, s=2, label="boundary")
    ax[0].set_aspect("equal"); ax[0].set_title("Input boundary"); ax[0].legend()

    ax[1].scatter(xB1, yB1, s=2, label="mapped boundary")
    circ_eq = patches.Circle((0,0), r_eq, fill=False, linestyle="--", linewidth=2, label="equal-area radius")
    circ_R  = patches.Circle((0,0), R,    fill=False, linewidth=1.5, label="final max radius")
    ax[1].add_patch(circ_eq); ax[1].add_patch(circ_R)
    ax[1].set_aspect("equal"); ax[1].set_title("Mapped boundary")
    ax[1].legend()

    plt.tight_layout(); plt.show()
'''