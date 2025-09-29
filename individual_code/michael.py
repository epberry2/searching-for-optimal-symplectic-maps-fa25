#!/usr/bin/env python3
# Boundary-only: learn a 2D symplectic map (composition of shears) that
# minimizes the maximum radius of the mapped boundary.

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import csv
import time

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
class ASymplectic:
    _x = sp.symbols('x')
    _default_factory = staticmethod(lambda D: [ASymplectic._x**m for m in range(D)])  # 1, x, x^2, ...
    _basis_factory = _default_factory

    @classmethod
    def configure_basis(cls, factory_or_name):
        if isinstance(factory_or_name, str):
            if factory_or_name.lower() == 'poly':
                cls._basis_factory = cls._default_factory
            else:
                raise ValueError("Unknown basis name for A: use 'poly' or pass a callable.")
        else:
            cls._basis_factory = staticmethod(factory_or_name)

    def __init__(self, coeffs):
        self.set_coeffs(coeffs)

    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        D = len(self.coeffs)
        x = self._x
        basis = ASymplectic._basis_factory(D)
        if len(basis) != D:
            raise ValueError(f"A-basis length {len(basis)} must equal number of coeffs {D}.")

        self.a_syms = sp.symbols(f'a0:{D}')
        f_expr = sum(self.a_syms[m] * basis[m] for m in range(D))

        f1 = sp.diff(f_expr, x)
        f2 = sp.diff(f1, x)
        J_list = [sp.diff(f1, p) for p in self.a_syms]

        self._f1 = sp.lambdify((x, *self.a_syms), f1, 'numpy')
        self._f2 = sp.lambdify((x, *self.a_syms), f2, 'numpy')
        self._J  = sp.lambdify((x, *self.a_syms), J_list, 'numpy')

    def __call__(self, x, y):
        return x, y + self._f1(x, *self.coeffs)

    def f2(self, x):
        out = self._f2(x, *self.coeffs)
        out = np.array(out, dtype=float)
        if out.shape == ():
            out = np.zeros_like(x, dtype=float) + float(out)
        return out

    def J_f1_params(self, x):
        raw = self._J(x, *self.coeffs) 
        cols = []
        for v in raw:
            v_arr = np.array(v, dtype=float)
            if v_arr.shape == (): 
                v_arr = np.zeros_like(x, dtype=float) + float(v_arr)
            cols.append(v_arr)
        vals = np.stack(cols, axis=1) 
        return vals


class BSymplectic:
    _y = sp.symbols('y')
    _default_factory = staticmethod(lambda D: [BSymplectic._y**m for m in range(D)])
    _basis_factory = _default_factory

    @classmethod
    def configure_basis(cls, factory_or_name):
        if isinstance(factory_or_name, str):
            if factory_or_name.lower() == 'poly':
                cls._basis_factory = cls._default_factory
            else:
                raise ValueError("Unknown basis name for B: use 'poly' or pass a callable.")
        else:
            cls._basis_factory = staticmethod(factory_or_name)

    # B_g: (x,y) -> (x + g'(y), y)
    def __init__(self, coeffs):
        self.set_coeffs(coeffs)

    def set_coeffs(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        D = len(self.coeffs)
        y = self._y
        basis = BSymplectic._basis_factory(D)
        if len(basis) != D:
            raise ValueError(f"B-basis length {len(basis)} must equal number of coeffs {D}.")

        self.b_syms = sp.symbols(f'b0:{D}')
        g_expr = sum(self.b_syms[m] * basis[m] for m in range(D))

        g1 = sp.diff(g_expr, y)
        g2 = sp.diff(g1, y)
        J_list = [sp.diff(g1, p) for p in self.b_syms]

        self._g1 = sp.lambdify((y, *self.b_syms), g1, 'numpy')
        self._g2 = sp.lambdify((y, *self.b_syms), g2, 'numpy')
        self._J  = sp.lambdify((y, *self.b_syms), J_list, 'numpy')

    def __call__(self, x, y):
        return x + self._g1(y, *self.coeffs), y

    def g2(self, y):
        out = self._g2(y, *self.coeffs)
        out = np.array(out, dtype=float)
        if out.shape == ():
            out = np.zeros_like(y, dtype=float) + float(out)
        return out

    def J_g1_params(self, y):
        raw = self._J(y, *self.coeffs)
        cols = []
        for v in raw:
            v_arr = np.array(v, dtype=float)
            if v_arr.shape == ():
                v_arr = np.zeros_like(y, dtype=float) + float(v_arr)
            cols.append(v_arr)
        vals = np.stack(cols, axis=1)   # (n_pts, D)
        return vals

def analytic_grad(Phi, xB, yB, w_center=1e-3, w_reg=5e-7):
    D = Phi.degree + 1
    theta = Phi.params()
    k = Phi.k

    preB_x, preB_y = [None]*k, [None]*k
    postB_x, postB_y = [None]*k, [None]*k
    postA_x, postA_y = [None]*k, [None]*k

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
        base = 2*i*D

        x_in_A = postB_x[i]
        J_a = Phi.A[i].J_f1_params(x_in_A)
        grad[base:base+D] += J_a.T @ gy
        gx = gx + gy * Phi.A[i].f2(x_in_A)

        # === B[i] backward ===
        y_in_B = preB_y[i]
        J_b = Phi.B[i].J_g1_params(y_in_B)  
        grad[base+D:base+2*D] += J_b.T @ gx
        gy = gy + gx * Phi.B[i].g2(y_in_B)

    grad += 2.0 * w_reg * theta
    return grad



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

if __name__ == "__main__":
    #This is example for other families of functions.
    # x, y = sp.symbols('x y')
    # ASymplectic.configure_basis(lambda D: [sp.sin(m*x) for m in range(D)])
    # BSymplectic.configure_basis(lambda D: [sp.sin(m*y) for m in range(D)])

    start = time.time()
    Phi, (xB0, yB0), (xB1, yB1), r_eq, hist, bestiter, frames = train_min_radius_boundary_2d(
        degree=5, k=7,
        n_boundary=5000, region=MultipleShapes([Rectangle([0, -0.5], verthalfwidth=0.25, horihalfwidth=0.5), Rectangle([0, 0.5], verthalfwidth=0.25, horihalfwidth=0.5)]),
        polynomial_bound=0.05,
        n_iters=5000, lr=2e-3, seed=10,
        w_center=1e-3, w_reg=5e-7, report_every=5,
        animate=True
    )
    end = time.time()
    elapsed = end - start
    print(f"\nTraining finished in {elapsed:.2f} seconds")
    
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
        ani.save("animations/test.mp4", writer="ffmpeg") # save as mp4, use pillow to save as gif

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

