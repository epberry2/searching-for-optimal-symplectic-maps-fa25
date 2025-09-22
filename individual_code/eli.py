#!/usr/bin/env python3
# Boundary-only: learn a 2D symplectic map (composition of shears) that
# minimizes the maximum radius of the mapped boundary.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


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

# ---------------------------
# Training (boundary only)
# ---------------------------
def train_min_radius_boundary_2d(
    degree=5, k=3,
    n_boundary=3000, vert=0.5, hori = 0.25,
    n_iters=300, lr=2e-2, seed=7,
    w_center=1e-3, w_reg=5e-7, report_every=25,
    animate=False
):
    rng = np.random.default_rng(seed)
    xB, yB = square_boundary_points(n_boundary, vert, hori, seed=seed)

    area = (2*vert) * (2*hori)
    r_eq = np.sqrt(area / np.pi)

    D = degree + 1
    num_params = 2*k*D
    theta0 = rng.uniform(-0.05, 0.05, num_params)
    Phi = SymplecticComposition(theta0, degree, k)
    opt = Adam(Phi.params(), lr=lr)

    def fobj(theta):
        Phi.set_params(theta)
        L, _, _ = loss_max_radius_boundary(Phi, xB, yB, w_center=w_center, w_reg=w_reg)
        return L

    history = []
    best = (np.inf, Phi.params())
    bestiter = 1

    # For animation
    frames = []

    for it in range(1, n_iters+1):
        xB, yB = square_boundary_points(n_boundary, vert, hori, seed= seed + it)
        grad = finite_diff_grad(fobj, Phi.params(), eps=1e-4)
        new_params = opt.step(Phi.params(), grad)
        Phi.set_params(new_params)

        L, aux, (xb, yb) = loss_max_radius_boundary(Phi, xB, yB, w_center=w_center, w_reg=w_reg)
        history.append(dict(it=it, loss=float(L), R=aux["R"], rmean=aux["rmean"],
                            rvar=aux["rvar"], cx=aux["cx"], cy=aux["cy"]))
        if L < best[0]:
            best = (float(L), Phi.params())
            bestiter = it

        if animate:
            frames.append((xb.copy(), yb.copy()))

        if it % report_every == 0 or it == 1 or it == n_iters:
            print(f"[{it:4d}] L={L:.6f}  R={aux['R']:.6f}  r_eq={r_eq:.6f}  "
                  f"rmean={aux['rmean']:.6f}  var={aux['rvar']:.3e}  "
                  f"cent=({aux['cx']:.2e},{aux['cy']:.2e})")

    xA, yA = square_boundary_points(n_boundary, vert, hori, seed= seed + bestiter)
    xB, yB = square_boundary_points(3000, vert, hori, seed= 0)
    Phi.set_params(best[1])
    xb, yb = Phi.forward(xB, yB)
    return Phi, (xB, yB), (xb, yb), (xA, yA), r_eq, history, frames if animate else None

# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    Phi, (xB0, yB0), (xB1, yB1), (xA0, yA0), r_eq, hist, frames = train_min_radius_boundary_2d(
        degree=5, k=5,
        n_boundary=3000, vert=0.5, hori=0.125,
        n_iters=300, lr=2e-2, seed=10,
        w_center=1e-3, w_reg=5e-7, report_every=25,
        animate=True
    )
    xA1, yA1 = Phi.forward(xA0, yA0)

    # final stats
    r = np.hypot(xB1, yB1)
    R = float(r.max())
    print("\nFinal summary:")
    print(f"  Equal-area lower bound r_eq = {r_eq:.6f}")
    print(f"  True max radius (boundary)  = {R:.6f}")
    print(f"  Mean boundary radius        = {float(r.mean()):.6f}")

      # Animation
    if frames is not None:
        fig, ax = plt.subplots()
        scat = ax.scatter([], [], s=2)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title("Boundary evolution")

        def update(frame):
            x, y = frame
            scat.set_offsets(np.column_stack([x, y]))
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
        ani.save("rect_evolution.gif", writer="pillow")
        plt.show()

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(xB0, yB0, s=2, label="boundary")
    #ax[0].scatter(xA0, yA0, c='red', s=10)
    ax[0].set_aspect("equal"); ax[0].set_title("Input boundary"); ax[0].legend()

    ax[1].scatter(xB1, yB1, s=2, label="mapped boundary")
    #ax[1].scatter(xA1, yA1, c='red', s=10)
    circ_eq = patches.Circle((0,0), r_eq, fill=False, linestyle="--", linewidth=2, label="equal-area radius")
    circ_R  = patches.Circle((0,0), R,    fill=False, linewidth=1.5, label="final max radius")
    ax[1].add_patch(circ_eq); ax[1].add_patch(circ_R)
    ax[1].set_aspect("equal"); ax[1].set_title("Mapped boundary")
    ax[1].legend()

    plt.tight_layout(); plt.show()
