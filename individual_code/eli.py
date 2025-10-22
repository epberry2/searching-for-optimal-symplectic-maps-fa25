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

# Optional PyTorch integration
try:
    import torch
except Exception:
    torch = None

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
        self._torch_phi = None
        self.set_params(params)

    def enable_torch(self, dtype=None, device=None):
        """Create an internal TorchSymplecticComposition synchronized with this object.
        After calling this, future set_params calls will also update the torch copy.
        """
        if torch is None:
            raise RuntimeError("PyTorch not available; cannot enable torch copy.")
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cpu')
        # create a torch Parameter that will be the backing storage for the torch phi
        theta_t = torch.nn.Parameter(torch.tensor(self._params, dtype=dtype, device=device))
        # build torch phi with param_tensor backing so updates to theta_t are visible
        self._torch_phi = TorchSymplecticComposition(param_tensor=theta_t, degree=self.degree, k=self.k, dtype=dtype, device=device)
        # also expose the parameter for external optimizers
        self._torch_param = theta_t

    def disable_torch(self):
        """Remove the internal torch copy."""
        self._torch_phi = None

    def params_tensor(self):
        """Return a torch tensor of the flat params if torch copy exists, else create one on CPU.
        This does not modify internal state.
        """
        if self._torch_phi is not None:
            return self._torch_phi.params(as_numpy=False)
        if torch is None:
            raise RuntimeError("PyTorch not available to create tensor params")
        return torch.tensor(self._params, dtype=torch.get_default_dtype())

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
        # keep torch copy synchronized if present
        if getattr(self, '_torch_phi', None) is not None:
            # convert numpy params to torch tensor with same dtype/device as torch_phi
            t = torch.tensor(self._params, dtype=self._torch_phi._params.dtype, device=self._torch_phi._params.device)
            self._torch_phi.set_flat_params(t)

    def params(self): return self._params.copy()

    def forward(self, x, y):
        for i in range(self.k-1, -1, -1):
            x, y = self.B[i](x, y)
            x, y = self.A[i](x, y)
        return x, y


class TorchSymplecticComposition:
    """Torch-native symplectic composition holding derivative coeffs in torch tensors.
    Params layout: flat 1D tensor length 2*k*D where D = degree+1, same ordering as numpy version.
    """
    def __init__(self, params=None, degree=None, k=None, dtype=None, device=None, param_tensor=None):
        if torch is None:
            raise RuntimeError("PyTorch is required for TorchSymplecticComposition")
        if param_tensor is None and (params is None or degree is None or k is None):
            raise ValueError("Provide either param_tensor or params+degree+k")
        self.degree = degree if degree is not None else int((param_tensor.numel() // 2)**0.5)
        self.k = k if k is not None else None
        if param_tensor is not None:
            t = param_tensor
            D = (t.numel() // (2 * (self.k if self.k is not None else 1)))
            # If degree provided, use that
            if degree is not None:
                D = degree + 1
        else:
            D = degree + 1
            if isinstance(params, np.ndarray):
                t = torch.tensor(params, dtype=dtype if dtype is not None else torch.get_default_dtype(), device=device)
            else:
                t = params.to(device=device, dtype=dtype) if isinstance(params, torch.Tensor) else torch.tensor(params, dtype=dtype, device=device)
        self.D = D
        # use the provided tensor as backing storage (no clone) so optimizer updates are visible
        self._params = t
        self._build_from_flat(self._params)

    def _build_from_flat(self, flat):
        D = self.D
        self.A_coeffs = []
        self.B_coeffs = []
        self.A_dcoeffs_desc = []
        self.B_dcoeffs_desc = []
        for i in range(self.k):
            start_a = 2*i*D
            start_b = 2*i*D + D
            a = flat[start_a : start_a + D]
            b = flat[start_b : start_b + D]
            # use views into flat so updates to flat are reflected
            self.A_coeffs.append(a)
            self.B_coeffs.append(b)
            # derivative coeffs (ascending) then reverse to descending for Horner
            if D > 1:
                idx = torch.arange(1, D, dtype=flat.dtype, device=flat.device)
                da = idx * a[1:D]
                db = idx * b[1:D]
                self.A_dcoeffs_desc.append(da.flip(0))
                self.B_dcoeffs_desc.append(db.flip(0))
            else:
                self.A_dcoeffs_desc.append(torch.tensor([], dtype=flat.dtype, device=flat.device))
                self.B_dcoeffs_desc.append(torch.tensor([], dtype=flat.dtype, device=flat.device))

    def set_flat_params(self, flat):
        # flat: torch tensor
        self._params = flat.clone().detach()
        self._build_from_flat(self._params)

    def params(self, as_numpy=False):
        """Return the flat parameter vector.
        by default returns a detached torch tensor on the same device/dtype as internal params.
        If as_numpy=True, returns a CPU numpy copy (matching the numpy `SymplecticComposition.params()` behavior).
        """
        t = self._params.clone().detach()
        if as_numpy:
            return t.cpu().numpy().copy()
        return t

    def forward(self, x, y):
        # x,y are torch tensors
        def polyval_desc(coeffs_desc, z):
            # coeffs_desc is descending order
            if coeffs_desc.numel() == 0:
                return torch.zeros_like(z)
            out = torch.zeros_like(z)
            for c in coeffs_desc:
                out = out * z + c
            return out

        for i in range(self.k-1, -1, -1):
            x = x + polyval_desc(self.B_dcoeffs_desc[i], y)
            y = y + polyval_desc(self.A_dcoeffs_desc[i], x)
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


def analytic_grad_torch(Phi, xB, yB, w_center=1e-3, w_reg=5e-7, eps_mask=0.0, dtype=None, device=None):
    """
    Torch implementation of analytic_grad. Returns a torch tensor gradient matching
    the flattened parameter vector in Phi (shape (num_params,)).
    Phi remains the numpy-based SymplecticComposition; we read its coeffs and
    perform the forward/backprop algebra using torch tensors so gradients can be
    applied with torch.optim.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for analytic_grad_torch but not available.")
    # choose dtype/device
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')

    # Support two Phi types: numpy-based SymplecticComposition and TorchSymplecticComposition
    is_torch_phi = hasattr(Phi, 'A_dcoeffs_desc') and hasattr(Phi, 'B_dcoeffs_desc')
    if is_torch_phi:
        D = Phi.degree + 1
        k = Phi.k
    else:
        D = Phi.degree + 1
        theta_np = Phi.params()
        k = Phi.k

    # Precompute coefficient tensors for the non-torch Phi to avoid repeated conversions
    if not is_torch_phi:
        A_desc_list = [None] * k
        B_desc_list = [None] * k
        if D >= 3:
            A_d2_desc_list = [None] * k
            B_d2_desc_list = [None] * k
        else:
            A_d2_desc_list = [torch.tensor([], dtype=dtype, device=device)] * k
            B_d2_desc_list = [torch.tensor([], dtype=dtype, device=device)] * k
        for i in range(k):
            A_desc_list[i] = torch.tensor(np.asarray(Phi.A[i].dcoeffs_desc, dtype=float), dtype=dtype, device=device)
            B_desc_list[i] = torch.tensor(np.asarray(Phi.B[i].dcoeffs_desc, dtype=float), dtype=dtype, device=device)
            if D >= 3:
                a = np.asarray(Phi.A[i].coeffs, dtype=float)
                d2_a = np.array([m*(m-1)*a[m] for m in range(2, D)], dtype=float)
                A_d2_desc_list[i] = torch.tensor(d2_a[::-1], dtype=dtype, device=device)
                b = np.asarray(Phi.B[i].coeffs, dtype=float)
                d2_b = np.array([m*(m-1)*b[m] for m in range(2, D)], dtype=float)
                B_d2_desc_list[i] = torch.tensor(d2_b[::-1], dtype=dtype, device=device)

    # helper: evaluate polynomial in descending coeff order via Horner
    def polyval_desc_torch(coeffs_desc, x):
        # coeffs_desc: 1D torch tensor [a_n,...,a_0]
        y = torch.zeros_like(x, dtype=dtype, device=device)
        for c in coeffs_desc:
            y = y * x + c
        return y

    # convert boundary to torch if necessary (avoid copies if already tensors)
    if isinstance(xB, torch.Tensor):
        x = xB.to(device=device, dtype=dtype)
    else:
        x = torch.tensor(xB, dtype=dtype, device=device)
    if isinstance(yB, torch.Tensor):
        y = yB.to(device=device, dtype=dtype)
    else:
        y = torch.tensor(yB, dtype=dtype, device=device)

    preB_x = [None] * k
    preB_y = [None] * k
    postB_x = [None] * k
    postB_y = [None] * k
    postA_x = [None] * k
    postA_y = [None] * k

    for i in range(k-1, -1, -1):
        preB_x[i], preB_y[i] = x, y
        if is_torch_phi:
            b_desc = Phi.B_dcoeffs_desc[i]
            x = x + polyval_desc_torch(b_desc, y)
            postB_x[i], postB_y[i] = x, y
            a_desc = Phi.A_dcoeffs_desc[i]
            y = y + polyval_desc_torch(a_desc, x)
            postA_x[i], postA_y[i] = x, y
        else:
            # B: x <- x + g'(y)
            b_desc = B_desc_list[i]
            x = x + polyval_desc_torch(b_desc, y)
            postB_x[i], postB_y[i] = x, y
            # A: y <- y + f'(x)
            a_desc = A_desc_list[i]
            y = y + polyval_desc_torch(a_desc, x)
            postA_x[i], postA_y[i] = x, y

    x_final, y_final = x, y

    r = torch.sqrt(x_final * x_final + y_final * y_final)
    R = float(r.max().cpu().detach().numpy())
    # create mask for near-max points
    mask = (r >= (R - eps_mask))

    gx = torch.zeros_like(x_final, dtype=dtype, device=device)
    gy = torch.zeros_like(y_final, dtype=dtype, device=device)
    # avoid division by zero
    denom = r[mask] + 1e-12
    if mask.any():
        gx[mask] = x_final[mask] / denom
        gy[mask] = y_final[mask] / denom

    gx = gx + 2.0 * w_center * (x_final.mean()) / x_final.numel()
    gy = gy + 2.0 * w_center * (y_final.mean()) / y_final.numel()

    # prepare grad tensor
    num_params = 2 * k * D
    grad = torch.zeros(num_params, dtype=dtype, device=device)
    for i in range(0, k):
        x_in_A = postB_x[i]
        base = 2*i*D
        if is_torch_phi:
            a = Phi.A_coeffs[i]
            # compute second-derivative descending coefficients: m*(m-1)*a[m]
            if D >= 3:
                idx = torch.arange(2, D, dtype=dtype, device=device)
                d2_a = idx * (idx - 1) * a[2:D]
                d2_a_desc = d2_a.flip(0)
                f2 = polyval_desc_torch(d2_a_desc, x_in_A)
            else:
                f2 = torch.zeros_like(x_in_A, dtype=dtype, device=device)
            for m in range(1, D):
                coeff = float(m)
                # a is ascending a0..a_{D-1}
                grad[base + m] = grad[base + m] + torch.sum(gy * (x_in_A ** (m-1)) * coeff)
            gx = gx + gy * f2
            y_in_B = preB_y[i]
            b = Phi.B_coeffs[i]
            if D >= 3:
                idx = torch.arange(2, D, dtype=dtype, device=device)
                d2_b = idx * (idx - 1) * b[2:D]
                d2_b_desc = d2_b.flip(0)
                g2 = polyval_desc_torch(d2_b_desc, y_in_B)
            else:
                g2 = torch.zeros_like(y_in_B, dtype=dtype, device=device)
            for m in range(1, D):
                coeff = float(m)
                grad[base + D + m] = grad[base + D + m] + torch.sum(gx * (y_in_B ** (m-1)) * coeff)
            gy = gy + gx * g2
        else:
            a = np.asarray(Phi.A[i].coeffs, dtype=float)
            if D >= 3:
                d2_a_desc = A_d2_desc_list[i]
                f2 = polyval_desc_torch(d2_a_desc, x_in_A)
            else:
                f2 = torch.zeros_like(x_in_A, dtype=dtype, device=device)
            for m in range(1, D):
                coeff = float(m)
                grad[base + m] = grad[base + m] + torch.sum(gy * (x_in_A ** (m-1)) * coeff)
            gx = gx + gy * f2
            y_in_B = preB_y[i]
            b = np.asarray(Phi.B[i].coeffs, dtype=float)
            if D >= 3:
                d2_b_desc = B_d2_desc_list[i]
                g2 = polyval_desc_torch(d2_b_desc, y_in_B)
            else:
                g2 = torch.zeros_like(y_in_B, dtype=dtype, device=device)
            for m in range(1, D):
                coeff = float(m)
                grad[base + D + m] = grad[base + D + m] + torch.sum(gx * (y_in_B ** (m-1)) * coeff)
            gy = gy + gx * g2

    # regularization term: prefer device-resident params when available
    if is_torch_phi and hasattr(Phi, 'params'):
        # Torch-backed phi: get tensor directly
        try:
            theta_torch = Phi.params(as_numpy=False)
        except Exception:
            theta_torch = Phi._params
        theta_torch = theta_torch.to(dtype=dtype, device=device)
    else:
        theta_torch = torch.tensor(Phi.params(), dtype=dtype, device=device)
    grad = grad + 2.0 * w_reg * theta_torch
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
    animate=False,
    use_torch=False,
    torch_dtype=None,
    torch_device=None,
    prealloc_pinned_boundary=False,
    sync_numpy_every=1,
    use_autocast=False,
    autocast_dtype=None,
    compile_grad=False,
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

    # Setup PyTorch optimizer if requested
    if use_torch:
        if torch is None:
            raise RuntimeError("PyTorch requested but not available. Install torch or set use_torch=False.")
        if torch_dtype is None:
            torch_dtype = torch.float32
        # choose device: prefer provided torch_device, else use CUDA if available
        if torch_device is None:
            torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # If using CUDA (A100), enable some performance flags where available
        if torch_device.type == 'cuda':
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
        # create a SymplecticComposition and enable its internal torch copy on chosen device
        Phi.enable_torch(dtype=torch_dtype, device=torch_device)
        # get backing param and optimizer
        theta_t = Phi._torch_param
        optim_torch = torch.optim.Adam([theta_t], lr=lr)
        torch_phi = Phi._torch_phi
        # set autocast dtype default for CUDA A100
        if use_autocast and autocast_dtype is None:
            autocast_dtype = torch.bfloat16 if torch_device.type == 'cuda' else torch.float16
        # optionally compile the analytic grad function to reduce Python overhead (PyTorch 2.x)
        if compile_grad and hasattr(torch, 'compile'):
            try:
                analytic_grad_torch_compiled = torch.compile(analytic_grad_torch)
            except Exception:
                analytic_grad_torch_compiled = analytic_grad_torch
        else:
            analytic_grad_torch_compiled = analytic_grad_torch

    history = []
    best = (np.inf, Phi.params())
    bestiter = 0

    # For animation
    frames = []

    # Optionally pre-allocate pinned host buffers for faster non-blocking H2D transfers
    if use_torch and prealloc_pinned_boundary:
        xB_pin = torch.empty(n_boundary, dtype=torch_dtype, pin_memory=True)
        yB_pin = torch.empty(n_boundary, dtype=torch_dtype, pin_memory=True)

    for it in range(1, n_iters+1):
        xB_np, yB_np = region.boundary_points(n_boundary, seed=seed + it)
        if use_torch:
            # Transfer boundary samples to device; use pinned-memory non-blocking copy when preallocated
            if prealloc_pinned_boundary:
                # handle variable-length boundary samples: resize pinned buffers if necessary
                Lb = xB_np.shape[0]
                if xB_pin.numel() < Lb:
                    xB_pin = torch.empty(Lb, dtype=torch_dtype, pin_memory=True)
                if yB_pin.numel() < Lb:
                    yB_pin = torch.empty(Lb, dtype=torch_dtype, pin_memory=True)
                # copy only the active slice into pinned host memory, then async transfer
                xB_pin_np = xB_pin.numpy()
                yB_pin_np = yB_pin.numpy()
                xB_pin_np[:Lb] = xB_np
                yB_pin_np[:Lb] = yB_np
                xB_t = xB_pin[:Lb].to(device=theta_t.device, non_blocking=True)
                yB_t = yB_pin[:Lb].to(device=theta_t.device, non_blocking=True)
            else:
                xB_t = torch.tensor(xB_np, dtype=torch_dtype, device=theta_t.device)
                yB_t = torch.tensor(yB_np, dtype=torch_dtype, device=theta_t.device)

            # compute torch-native analytic gradient and step on torch_phi
            # Optionally run analytic grad under autocast to leverage Tensor Cores
            if use_autocast and torch_device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    grad_t = analytic_grad_torch_compiled(torch_phi, xB_t, yB_t, w_center=w_center, w_reg=w_reg,
                                                          eps_mask=0.0, dtype=torch_dtype, device=theta_t.device)
            else:
                grad_t = analytic_grad_torch_compiled(torch_phi, xB_t, yB_t, w_center=w_center, w_reg=w_reg,
                                                      eps_mask=0.0, dtype=torch_dtype, device=theta_t.device)

            # optimizer expects float32 grads for stability; cast if necessary
            with torch.no_grad():
                optim_torch.zero_grad()
                if grad_t.dtype != theta_t.dtype:
                    theta_t.grad = grad_t.to(dtype=theta_t.dtype)
                else:
                    theta_t.grad = grad_t
                optim_torch.step()

            # sync numpy params only every sync_numpy_every iterations to reduce overhead
            if sync_numpy_every is not None and sync_numpy_every > 0 and (it % sync_numpy_every == 0):
                Phi.set_params(theta_t.detach().cpu().numpy().astype(float))

            # Evaluate mapped boundary for logging without grad tracking
            with torch.no_grad():
                xb_t, yb_t = torch_phi.forward(xB_t, yB_t)
                r_t = torch.sqrt(xb_t * xb_t + yb_t * yb_t)
                L = float(r_t.max().cpu().detach().numpy()) + w_center * (float(xb_t.mean().cpu().detach().numpy())**2 + float(yb_t.mean().cpu().detach().numpy())**2) + w_reg * float((theta_t.detach()**2).sum().cpu().numpy())
                aux = dict(R=float(r_t.max().cpu().detach().numpy()), cx=float(xb_t.mean().cpu().detach().numpy()), cy=float(yb_t.mean().cpu().detach().numpy()), rmean=float(r_t.mean().cpu().detach().numpy()), rvar=float(r_t.var().cpu().detach().numpy()))
                xb = xb_t.detach().cpu().numpy(); yb = yb_t.detach().cpu().numpy()
        else:
            grad = analytic_grad(Phi, xB_np, yB_np, w_center=w_center, w_reg=w_reg)
            new_params = opt.step(Phi.params(), grad)
            Phi.set_params(new_params)
            L, aux, (xb, yb) = loss_max_radius_boundary(Phi, xB_np, yB_np, w_center=w_center, w_reg=w_reg)

        # check for non-finite loss 
        if not np.isfinite(L):
            print(f"[{it:4d}] Non-finite loss encountered (L={L}); stopping.")
            break

        history.append(dict(it=it, loss=float(L), R=aux["R"], rmean=aux["rmean"],
                            rvar=aux["rvar"], cx=aux["cx"], cy=aux["cy"]))

        if L < best[0]:
            # If we're using the torch path and not syncing numpy params every iter,
            # Phi.params() may be stale (still initial). Use the torch backing param
            # as the authoritative source when available.
            if use_torch and 'theta_t' in locals():
                best_params = theta_t.detach().cpu().numpy().astype(float)
            else:
                best_params = Phi.params()
            best = (float(L), best_params)
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
    Phi, (xB0, yB0), (xB1, yB1), r_eq, hist, bestiter, frames = train_min_radius_boundary_2d(
        degree=5, k=10,
        n_boundary=5000,
        region=Circle([0.0, 1.0], 1.0),
        polynomial_bound=0.005,
        n_iters=500, lr=2e-3, seed=10,
        w_center=1e-3, w_reg=5e-7, report_every=50,
        animate=True,
        use_torch=True,
        torch_dtype=torch.float32,
        torch_device=None,
        prealloc_pinned_boundary=False,
        sync_numpy_every=100
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
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title("Boundary evolution")

        def update(frame):
            x, y = frame
            scat.set_offsets(np.column_stack([x, y]))
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
        ani.save("test1.gif", writer="pillow") # use ffmpeg to save as mp4, use pillow to save as gif

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
