import numpy as np
from numpy.polynomial.polynomial import polyval2d as _np_polyval2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, csv, argparse

def _polyval2d_desc(x, y, c_desc):
    c_asc = c_desc[::-1, ::-1]
    return _np_polyval2d(x, y, c_asc)

np.polyval2d = _polyval2d_desc



class Shape4D:
    def volume(self): return 1.0
    def boundary_points(self, k=1, seed=0): raise NotImplementedError

class EllipsoidE1a(Shape4D):
    """
    Boundary sampler for the coupled ellipsoid on the blackboard:
      E(1,a) = { x1^2 + y1^2 + (x2^2 + y2^2)/a <= 1 }.
    Volume(E(1,a)) = (pi^2/2) * a.
    """
    def __init__(self, a=1.0):
        assert a > 0, "a must be positive"
        self.a = float(a)

    def boundary_points(self, k=6000, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0.0, np.pi/2.0, size=k)
        alpha = rng.uniform(0.0, 2*np.pi, size=k)
        beta  = rng.uniform(0.0, 2*np.pi, size=k)
        r1 = np.cos(t)            
        r2 = np.sqrt(self.a) * np.sin(t) 

        x1 = r1 * np.cos(alpha)
        y1 = r1 * np.sin(alpha)
        x2 = r2 * np.cos(beta)
        y2 = r2 * np.sin(beta)
        return x1, x2, y1, y2

    def volume(self):
        return (np.pi**2 / 2.0) * self.a

class Ellipsoid4D(Shape4D):
    def __init__(self, position=(0,0,0,0), radii=(1,1,1,1)):
        self.position = np.asarray(position, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
    def boundary_points(self, k=1, seed=0):
        rng = np.random.default_rng(seed)
        u = rng.normal(0,1,(4,k))
        u /= np.linalg.norm(u, axis=0, keepdims=True)
        pts = (self.radii[:,None] * u) + self.position[:,None]
        return pts[0], pts[1], pts[2], pts[3]
    def volume(self):
        return (np.pi**2/2.0) * float(np.prod(self.radii))

class LagrangianTorus4D(Shape4D):
    def __init__(self, center=(0,0,0,0), radii_xy1=(1,1), radii_xy2=(1,1)):
        self.center = np.asarray(center, dtype=float)
        self.rxy1 = np.asarray(radii_xy1, dtype=float)
        self.rxy2 = np.asarray(radii_xy2, dtype=float)
    def boundary_points(self, k=2048, seed=0):
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0,2*np.pi,k)
        t2 = rng.uniform(0,2*np.pi,k)
        x1 = self.center[0] + self.rxy1[0]*np.cos(t1)
        y1 = self.center[1] + self.rxy1[1]*np.sin(t1)
        x2 = self.center[2] + self.rxy2[0]*np.cos(t2)
        y2 = self.center[3] + self.rxy2[1]*np.sin(t2)
        return x1, x2, y1, y2

class Union4D(Shape4D):
    def __init__(self, shapes):
        self.shapes = list(shapes)
    def boundary_points(self, k=4000, seed=0):
        xs1=[]; xs2=[]; ys1=[]; ys2=[]
        per = max(1, k//len(self.shapes))
        for i,sh in enumerate(self.shapes):
            x1,x2,y1,y2 = sh.boundary_points(per, seed=seed+11*i)
            xs1.append(x1); xs2.append(x2); ys1.append(y1); ys2.append(y2)
        return np.concatenate(xs1), np.concatenate(xs2), np.concatenate(ys1), np.concatenate(ys2)
    def volume(self):
        return sum(getattr(s,'volume',lambda:0.0)() for s in self.shapes)


def poly2d_second_derivs(coeffs):
    c = np.asarray(coeffs, dtype=float)
    D = c.shape[0]
    assert c.shape[0]==c.shape[1]
    p_xx = np.zeros((max(D-2,1), D))
    for i in range(2, D):
        p_xx[i-2,:] = i*(i-1)*c[i,:]
    p_yy = np.zeros((D, max(D-2,1)))
    for j in range(2, D):
        p_yy[:,j-2] = j*(j-1)*c[:,j]
    p_xy = np.zeros((max(D-1,1), max(D-1,1)))
    if D>=2:
        for i in range(1,D):
            for j in range(1,D):
                p_xy[i-1,j-1] = i*j*c[i,j]
    def desc(M): return M[::-1, ::-1] if M.size>1 else M
    return desc(p_xx), desc(p_xy), desc(p_yy)



class ASymplecticR4:
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim==2 and c.shape[0]==c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D-1, D)) if D>1 else np.zeros((1, D))
        d2 = np.zeros((D, D-1)) if D>1 else np.zeros((D, 1))
        for i in range(1,D): d1[i-1,:] = i*c[i,:] 
        for j in range(1,D): d2[:,j-1] = j*c[:,j]
        self.d1_desc = d1[::-1, ::-1] if d1.size>1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size>1 else d2
        p_xx, p_xy, p_yy = poly2d_second_derivs(c)
        self.f_xx_desc, self.f_xy_desc, self.f_yy_desc = p_xx, p_xy, p_yy
    def __call__(self, x1,x2,y1,y2):
        y1 = y1 + np.polyval2d(x1, x2, self.d1_desc)
        y2 = y2 + np.polyval2d(x1, x2, self.d2_desc)
        return x1,x2,y1,y2

class BSymplecticR4:
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim==2 and c.shape[0]==c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D-1, D)) if D>1 else np.zeros((1, D))
        d2 = np.zeros((D, D-1)) if D>1 else np.zeros((D, 1))
        for i in range(1,D): d1[i-1,:] = i*c[i,:] 
        for j in range(1,D): d2[:,j-1] = j*c[:,j] 
        self.d1_desc = d1[::-1, ::-1] if d1.size>1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size>1 else d2
        p_xx, p_xy, p_yy = poly2d_second_derivs(c)   
        self.g_11_desc, self.g_12_desc, self.g_22_desc = p_xx, p_xy, p_yy
    def __call__(self, x1,x2,y1,y2):
        x1 = x1 + np.polyval2d(y1, y2, self.d1_desc)
        x2 = x2 + np.polyval2d(y1, y2, self.d2_desc)
        return x1,x2,y1,y2

class CSymplecticR4:
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim == 2 and c.shape[0] == c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D-1, D)) if D > 1 else np.zeros((1, D))
        d2 = np.zeros((D, D-1)) if D > 1 else np.zeros((D, 1))
        for i in range(1, D): d1[i-1, :] = i * c[i, :]
        for j in range(1, D): d2[:, j-1] = j * c[:, j]
        self.d1_desc = d1[::-1, ::-1] if d1.size > 1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size > 1 else d2
        p_xx, p_xy, p_yy = poly2d_second_derivs(c)
        self.c_11_desc, self.c_12_desc, self.c_22_desc = p_xx, p_xy, p_yy
    def __call__(self, x1, x2, y1, y2):
        y1 = y1 - np.polyval2d(x1, y2, self.d1_desc)
        x2 = x2 + np.polyval2d(x1, y2, self.d2_desc)
        return x1, x2, y1, y2

class DSymplecticR4:
    def __init__(self, coeffs): self.set_coeffs(coeffs)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim == 2 and c.shape[0] == c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D-1, D)) if D > 1 else np.zeros((1, D))
        d2 = np.zeros((D, D-1)) if D > 1 else np.zeros((D, 1))
        for i in range(1, D): d1[i-1, :] = i * c[i, :]
        for j in range(1, D): d2[:, j-1] = j * c[:, j]
        self.d1_desc = d1[::-1, ::-1] if d1.size > 1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size > 1 else d2
        p_xx, p_xy, p_yy = poly2d_second_derivs(c)
        self.d_11_desc, self.d_12_desc, self.d_22_desc = p_xx, p_xy, p_yy
    def __call__(self, x1, x2, y1, y2):
        x1 = x1 + np.polyval2d(y1, x2, self.d1_desc)
        y2 = y2 - np.polyval2d(y1, x2, self.d2_desc)
        return x1, x2, y1, y2

class TuraevSymplecticR4:
    def __init__(self, coeffs, C=(0.0, 0.0)):
        self.set_coeffs(coeffs)
        self.C = np.asarray(C, dtype=float)
    def set_coeffs(self, coeffs):
        c = np.asarray(coeffs, dtype=float)
        assert c.ndim == 2 and c.shape[0] == c.shape[1]
        self.coeffs = c
        D = c.shape[0]
        d1 = np.zeros((D - 1, D)) if D > 1 else np.zeros((1, D))
        d2 = np.zeros((D, D - 1)) if D > 1 else np.zeros((D, 1))
        for i in range(1, D):
            d1[i - 1, :] = i * c[i, :]
        for j in range(1, D):
            d2[:, j - 1] = j * c[:, j]
        self.d1_desc = d1[::-1, ::-1] if d1.size > 1 else d1
        self.d2_desc = d2[::-1, ::-1] if d2.size > 1 else d2
        p_yy11, p_yy12, p_yy22 = poly2d_second_derivs(c)
        self.v_11_desc, self.v_12_desc, self.v_22_desc = p_yy11, p_yy12, p_yy22
    def __call__(self, x1, x2, y1, y2):
        Vy1 = np.polyval2d(y1, y2, self.d1_desc)
        Vy2 = np.polyval2d(y1, y2, self.d2_desc)
        x1_new = y1 + self.C[0]
        x2_new = y2 + self.C[1]
        y1_new = -x1 + Vy1
        y2_new = -x2 + Vy2
        return x1_new, x2_new, y1_new, y2_new




class SymplecticCompositionR4:
    def __init__(self, params, degree, k, use_turaev=False):
        self.degree = degree
        self.k = k
        self.use_turaev = use_turaev
        self.set_params(params)
    def set_params(self, params):
        params = np.asarray(params, dtype=float)
        D = self.degree + 1
        need = 2*self.k*D*D
        assert params.size==need, f"need {need} params, got {params.size}"
        self._params = params.copy()
        self.A=[]; self.B=[]
        off=0
        for _ in range(self.k):
            a = params[off:off+D*D].reshape(D,D); off += D*D
            b = params[off:off+D*D].reshape(D,D); off += D*D
            self.A.append(ASymplecticR4(a))
            self.B.append(BSymplecticR4(b))
            if self.use_turaev:
                self.T.append(TuraevSymplecticR4(a, C=(0.0, 0.0)))
    def params(self): return self._params.copy()
    def set_params_inplace(self, p): self._params[:] = p; self.set_params(p)  
    def forward(self, x1,x2,y1,y2):
        for i in range(self.k-1, -1, -1):
            x1,x2,y1,y2 = self.B[i](x1,x2,y1,y2)
            x1,x2,y1,y2 = self.A[i](x1,x2,y1,y2)
            if self.use_turaev:
                x1, x2, y1, y2 = self.T[i](x1, x2, y1, y2)
        return x1,x2,y1,y2

def loss_max_radius_boundary_R4(Phi, x1B,x2B,y1B,y2B, w_center=1e-3, w_reg=1e-7, tau=None):
    X1,X2,Y1,Y2 = Phi.forward(x1B,x2B,y1B,y2B)
    r = np.sqrt(X1*X1 + X2*X2 + Y1*Y1 + Y2*Y2)
    R = float(r.max())
    cx = float(X1.mean()); cy = float(X2.mean()); cp = float(Y1.mean()); cq = float(Y2.mean())
    reg = float(np.sum(Phi.params()**2))
    # smooth max option: use log-sum-exp approximation when tau is provided (>0)
    if tau is None or tau == 0:
        Lmax = R
    else:
        t = tau * r
        tmax = float(t.max())
        s = float(np.exp(t - tmax).sum())
        Lmax = (tmax + np.log(s)) / float(tau)
    L = Lmax + w_center*(cx*cx + cy*cy + cp*cp + cq*cq) + w_reg*reg
    aux = dict(R=R, R_smooth=float(Lmax), rmean=float(r.mean()), rvar=float(r.var()), cx=cx, cy=cy, cp=cp, cq=cq)
    return L, aux, (X1,X2,Y1,Y2,r)

def analytic_grad_R4(Phi, x1B,x2B,y1B,y2B, w_center=1e-3, w_reg=1e-7, tau=None):
    k = Phi.k; D = Phi.degree+1
    preB = [None]*k
    postB = [None]*k
    postA = [None]*k
    x1,x2,y1,y2 = x1B, x2B, y1B, y2B
    for i in range(k-1, -1, -1):
        preB[i]  = (x1,x2,y1,y2)
        x1,x2,y1,y2 = Phi.B[i](x1,x2,y1,y2)
        postB[i] = (x1,x2,y1,y2)
        x1,x2,y1,y2 = Phi.A[i](x1,x2,y1,y2)
        postA[i] = (x1,x2,y1,y2)
    X1,X2,Y1,Y2 = x1,x2,y1,y2
    r2 = X1*X1 + X2*X2 + Y1*Y1 + Y2*Y2
    r = np.sqrt(r2)
    R = r.max()
    eps = 1e-12
    n = X1.size
    # base derivatives: either hard-max (mask) or softmax weights when tau>0
    if tau is None or tau == 0:
        mask = (r >= R - 0.0)
        dL_dX1 = np.zeros_like(X1); dL_dX2 = np.zeros_like(X2)
        dL_dY1 = np.zeros_like(Y1); dL_dY2 = np.zeros_like(Y2)
        dL_dX1[mask] = X1[mask] / (r[mask] + eps)
        dL_dX2[mask] = X2[mask] / (r[mask] + eps)
        dL_dY1[mask] = Y1[mask] / (r[mask] + eps)
        dL_dY2[mask] = Y2[mask] / (r[mask] + eps)
    else:
        t = tau * r
        tmax = float(t.max())
        e = np.exp(t - tmax)
        w = e / (e.sum() + 1e-30)
        dL_dX1 = w * (X1 / (r + eps))
        dL_dX2 = w * (X2 / (r + eps))
        dL_dY1 = w * (Y1 / (r + eps))
        dL_dY2 = w * (Y2 / (r + eps))

    # center-term added uniformly across samples
    dL_dX1 += 2.0*w_center*(X1.mean())/n
    dL_dX2 += 2.0*w_center*(X2.mean())/n
    dL_dY1 += 2.0*w_center*(Y1.mean())/n
    dL_dY2 += 2.0*w_center*(Y2.mean())/n
    grad = np.zeros_like(Phi.params())
    D2 = D*D
    layer_offsets = [(2*i*D2, 2*i*D2 + D2, 2*i*D2 + 2*D2) for i in range(Phi.k)]
    gx1, gx2, gy1, gy2 = dL_dX1, dL_dX2, dL_dY1, dL_dY2
    for i in range(k-1, -1, -1):
        a_start, b_start, _ = layer_offsets[i]
        x1_in, x2_in, y1_in, y2_in = postB[i]
        # A-layer coeff grads
        X1p = np.vstack([x1_in**p for p in range(D)])
        X2q = np.vstack([x2_in**q for q in range(D)])
        term1 = np.zeros((D,D)); term2 = np.zeros((D,D))
        for p in range(1,D):
            for q in range(D):
                term1[p,q] = np.sum(gy1 * (p * X1p[p-1] * X2q[q]))
        for p in range(D):
            for q in range(1,D):
                term2[p,q] = np.sum(gy2 * (q * X1p[p] * X2q[q-1]))
        grad[a_start:a_start+D2] += (term1 + term2).ravel()
        f_xx = np.polyval2d(x1_in, x2_in, Phi.A[i].f_xx_desc)
        f_xy = np.polyval2d(x1_in, x2_in, Phi.A[i].f_xy_desc)
        f_yy = np.polyval2d(x1_in, x2_in, Phi.A[i].f_yy_desc)
        gx1 = gx1 + gy1 * f_xx + gy2 * f_xy
        gx2 = gx2 + gy1 * f_xy + gy2 * f_yy
        x1_pre, x2_pre, y1_pre, y2_pre = preB[i]
        Y1p = np.vstack([y1_pre**p for p in range(D)])
        Y2q = np.vstack([y2_pre**q for q in range(D)])
        term1 = np.zeros((D,D)); term2 = np.zeros((D,D))
        for p in range(1,D):
            for q in range(D):
                term1[p,q] = np.sum(gx1 * (p * Y1p[p-1] * Y2q[q]))
        for p in range(D):
            for q in range(1,D):
                term2[p,q] = np.sum(gx2 * (q * Y1p[p] * Y2q[q-1]))
        grad[b_start:b_start+D2] += (term1 + term2).ravel()
        g_11 = np.polyval2d(y1_pre, y2_pre, Phi.B[i].g_11_desc)
        g_12 = np.polyval2d(y1_pre, y2_pre, Phi.B[i].g_12_desc)
        g_22 = np.polyval2d(y1_pre, y2_pre, Phi.B[i].g_22_desc)
        gy1 = gy1 + gx1 * g_11 + gx2 * g_12
        gy2 = gy2 + gx1 * g_12 + gx2 * g_22
    grad += 2.0*w_reg * Phi.params()
    return grad

class Adam:
    def __init__(self, params, lr=2e-3, b1=0.5, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=np.zeros_like(params); self.v=np.zeros_like(params); self.t=0
    def step(self, params, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*(grad*grad)
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return params - self.lr * mhat / (np.sqrt(vhat) + self.eps)



def train_min_radius_boundary_R4(
    degree=3, k=6,
    n_boundary=6000,
    region=EllipsoidE1a(a=1.44),
    n_iters=300, lr=1e-3, seed=11,
    polynomial_bound=5e-3,
    w_center=1e-3, w_reg=1e-7, report_every=25,
    tau=None,
    clip_grad_norm=None,
    max_step_retries=5,
    lr_backoff_factor=0.1,
    min_lr=1e-12,
    reset_momentum_on_reject=True,
    revert_to_best_on_failure=True,
    animate=True, max_frames=60
):
    rng = np.random.default_rng(seed)
    D = degree + 1
    num_params = 2*k*D*D
    theta0 = rng.uniform(-polynomial_bound, polynomial_bound, num_params)
    Phi = SymplecticCompositionR4(theta0, degree, k)

    #Add Turave 
    turaev_coeffs = rng.uniform(-5e-3, 5e-3, (degree+1, degree+1))
    Phi.turaev = TuraevSymplecticR4(turaev_coeffs, C=(0.0, 0.0))
    #---------

    opt = Adam(Phi.params(), lr=lr)

    history = []
    best = (np.inf, Phi.params())
    bestiter = 0
    frames = []
    x1B,x2B,y1B,y2B = region.boundary_points(n_boundary, seed=seed)
    for it in range(1, n_iters+1):
        #x1B,x2B,y1B,y2B = region.boundary_points(n_boundary, seed=seed+it)

        # compute analytic gradient
        grad = analytic_grad_R4(Phi, x1B,x2B,y1B,y2B, w_center=w_center, w_reg=w_reg, tau=tau)

        # ---- compute numerical gradient for Turaev layer ----
        grad_turaev = None
        if hasattr(Phi, "turaev") and Phi.turaev is not None:
            eps = 1e-4
            c = Phi.turaev.coeffs.copy()
            dL_dc = np.zeros_like(c)
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    c_perturb = c.copy()
                    c_perturb[i, j] += eps
                    Phi.turaev.set_coeffs(c_perturb)
                    L_plus, _, _ = loss_max_radius_boundary_R4(Phi, x1B, x2B, y1B, y2B, w_center, w_reg, tau)
                    c_perturb[i, j] -= 2 * eps
                    Phi.turaev.set_coeffs(c_perturb)
                    L_minus, _, _ = loss_max_radius_boundary_R4(Phi, x1B, x2B, y1B, y2B, w_center, w_reg, tau)
                    dL_dc[i, j] = (L_plus - L_minus) / (2 * eps)
            Phi.turaev.set_coeffs(c)  # restore
            grad_turaev = dL_dc
        # -------------------------------------------------------

        if not np.all(np.isfinite(grad)):
            print(f"[{it:4d}] Non-finite gradient; stopping."); break

        # record grad norm for diagnostics and optionally clip
        gnorm = float(np.linalg.norm(grad))
        if clip_grad_norm is not None and gnorm > 0 and gnorm > clip_grad_norm:
            grad = grad * (float(clip_grad_norm) / (gnorm + 1e-12))
            #print(f"[{it:4d}] Clipped grad norm {gnorm:.3e} -> {float(np.linalg.norm(grad)):.3e}")

        # compute current loss before stepping (used for rollback decisions)
        L_prev, aux_prev, _ = loss_max_radius_boundary_R4(Phi, x1B,x2B,y1B,y2B, w_center=w_center, w_reg=w_reg, tau=tau)

        # attempt step(s) with rollback on bad loss and LR reduction
        attempts = 0
        accepted = False
        while attempts <= max_step_retries:
            old_params = Phi.params().copy()
            new_params = opt.step(old_params, grad)
            if not np.all(np.isfinite(new_params)):
                print(f"[{it:4d}] Non-finite parameters from optimizer; reducing lr and retrying.")
                opt.lr *= float(lr_backoff_factor)
                if reset_momentum_on_reject:
                    opt.m[:] = 0.0; opt.v[:] = 0.0; opt.t = 0
                    print(f"[{it:4d}] Reset optimizer momentum (m,v,t).")
                attempts += 1
                if opt.lr < min_lr:
                    print(f"[{it:4d}] Learning rate dropped below min_lr={min_lr:.1e}; stopping.")
                    if revert_to_best_on_failure:
                        Phi.set_params(best[1])
                    accepted = False
                    break
                continue
            Phi.set_params(new_params)
            #train Turaev
            if grad_turaev is not None:
                Phi.turaev.coeffs -= opt.lr * grad_turaev
            L_new, aux_new, (X1,X2,Y1,Y2,r) = loss_max_radius_boundary_R4(Phi, x1B,x2B,y1B,y2B, w_center=w_center, w_reg=w_reg, tau=tau)
            if not np.isfinite(L_new) or L_new > L_prev * 1.25 + 1e-12:
                # reject step, back off lr and retry
                print(f"[{it:4d}] Rejected step: L_prev={L_prev:.6e} L_new={L_new:.6e}; reducing lr and reverting params.")
                Phi.set_params(old_params)
                opt.lr *= float(lr_backoff_factor)
                if reset_momentum_on_reject:
                    opt.m[:] = 0.0; opt.v[:] = 0.0; opt.t = 0
                    print(f"[{it:4d}] Reset optimizer momentum (m,v,t) after rejected step.")
                attempts += 1
                if opt.lr < min_lr:
                    print(f"[{it:4d}] Learning rate dropped below min_lr={min_lr:.1e}; reverting to best and stopping.")
                    if revert_to_best_on_failure:
                        Phi.set_params(best[1])
                    accepted = False
                    break
                continue
            # accepted
            accepted = True
            L, aux = L_new, aux_new
            break

        if not accepted:
            # if all retries failed, stop training to avoid runaway
            print(f"[{it:4d}] Failed to find acceptable step after {attempts} attempts; stopping.")
            break
        history.append(dict(it=it, loss=float(L), grad_norm=gnorm, **aux))
        if L < best[0]:
            best = (float(L), Phi.params().copy())
            bestiter = it

        if animate and (it % report_every == 0 or it==1 or it==n_iters) and len(frames) < max_frames:
            take = min(4000, X1.size)
            idx = np.random.default_rng(1234).choice(X1.size, size=take, replace=False)
            frames.append((X1[idx], Y1[idx], X2[idx], Y2[idx], aux['R']))

        if it % report_every == 0 or it==1 or it==n_iters:
            print(f"[{it:4d}] L={L:.6f}  R={aux['R']:.6f}  rmean={aux['rmean']:.6f}  var={aux['rvar']:.3e}  "
                  f"cent=({aux['cx']:.2e},{aux['cy']:.2e},{aux['cp']:.2e},{aux['cq']:.2e})")

    Phi.set_params(best[1])
    x1B,x2B,y1B,y2B = region.boundary_points(n_boundary, seed=seed+9999)
    X1,X2,Y1,Y2 = Phi.forward(x1B,x2B,y1B,y2B)
    return Phi, (x1B,x2B,y1B,y2B), (X1,X2,Y1,Y2), history, bestiter, frames



def save_projection_animation(frames, outpath="r4_training.gif"):
    if not frames:
        print("No frames to animate."); return
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    scat1 = axes[0].scatter([], [], s=1)
    scat2 = axes[1].scatter([], [], s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-3,3); ax.set_ylim(-3,3)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    def update(frame):
        X1,Y1,X2,Y2,R = frame
        scat1.set_offsets(np.column_stack([X1, Y1]))
        scat2.set_offsets(np.column_stack([X2, Y2]))
        axes[0].set_title(f"(x1,y1)  R≈{R:.3f}")
        axes[1].set_title(f"(x2,y2)  R≈{R:.3f}")
        return scat1, scat2
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="ffmpeg")
    plt.close(fig)
    print(f"Saved animation to {outpath}")


def save_radial_projection_animation(frames, outpath="r4_radial_training.gif"):
    """Save an animation projecting 4D points to the plane
    (x1^2 + y1^2, x2^2 + y2^2).
    Expects frames as a list of tuples: (X1, Y1, X2, Y2, R)
    where X1 etc are 1D numpy arrays of the same length for that frame.
    """
    if not frames:
        print("No frames to animate."); return
    # compute reasonable axis limits from pooled data (small number of frames)
    all_r1_min = np.inf; all_r1_max = -np.inf
    all_r2_min = np.inf; all_r2_max = -np.inf
    for X1, Y1, X2, Y2, _ in frames:
        r1 = 3 * np.pi * (X1**2 + Y1**2)
        r2 = np.pi * (X2**2 + Y2**2)
        if r1.size:
            all_r1_min = min(all_r1_min, float(r1.min())); all_r1_max = max(all_r1_max, float(r1.max()))
        if r2.size:
            all_r2_min = min(all_r2_min, float(r2.min())); all_r2_max = max(all_r2_max, float(r2.max()))

    pad1 = 0.05 * (all_r1_max - all_r1_min) if all_r1_max>all_r1_min else 0.1
    pad2 = 0.05 * (all_r2_max - all_r2_min) if all_r2_max>all_r2_min else 0.1

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    scat = ax.scatter([], [], s=1)
    ax.set_aspect('equal')
    ax.set_xlim(all_r1_min - pad1, all_r1_max + pad1)
    ax.set_ylim(all_r2_min - pad2, all_r2_max + pad2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Radial projection (x1^2+y1^2 vs x2^2+y2^2)")

    def update(frame):
        X1, Y1, X2, Y2, R = frame
        r1 = 3 * np.pi * (X1**2 + Y1**2)
        r2 = np.pi * (X2**2 + Y2**2)
        coords = np.column_stack([r1, r2])
        scat.set_offsets(coords)
        ax.set_title(f"Radial projection  R≈{R:.3f}")
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="ffmpeg")
    plt.close(fig)
    print(f"Saved radial projection animation to {outpath}")



def build_region(args):
    if args.region == "E1a":
        return EllipsoidE1a(a=args.a)
    elif args.region == "ellipsoid":
        r = tuple(map(float, args.radii.split(",")))
        assert len(r)==4, "--radii must have 4 comma-separated floats"
        return Ellipsoid4D(radii=r)
    elif args.region == "torus":
        r1 = tuple(map(float, args.rxy1.split(",")))
        r2 = tuple(map(float, args.rxy2.split(",")))
        return LagrangianTorus4D(radii_xy1=r1, radii_xy2=r2)
    elif args.region == "union_tori":
        r1 = tuple(map(float, args.rxy1.split(",")))
        r2 = tuple(map(float, args.rxy2.split(",")))
        t1 = LagrangianTorus4D(center=(-0.4,0,0,0), radii_xy1=r1, radii_xy2=r2)
        t2 = LagrangianTorus4D(center=( 0.4,0,0,0), radii_xy1=r1, radii_xy2=r2)
        return Union4D([t1,t2])
    else:
        raise ValueError(f"Unknown region {args.region}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-iters", type=int, default=30000)
    ap.add_argument("--n-boundary", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--region", type=str, default="ellipsoid",
                    choices=["E1a","ellipsoid","torus","union_tori"])
    ap.add_argument("--a", type=float, default="2.0", help="parameter a in E(1,a)")

    ap.add_argument("--radii", type=str, default="1,3,1,3")
    ap.add_argument("--rxy1", type=str, default="0.5,0.5")
    ap.add_argument("--rxy2", type=str, default="0.8,0.6")

    ap.add_argument("--outdir", type=str, default="output_r4")
    ap.add_argument("--anim", action="store_true", default=True)
    args = ap.parse_args()

    region = build_region(args)
    rng = np.random.default_rng(args.seed)
    D = args.degree + 1
    num_params = 2*args.k*D*D
    theta0 = rng.uniform(-5e-3, 5e-3, num_params)
    Phi = SymplecticCompositionR4(theta0, args.degree, args.k)

    Phi, startB, endB, hist, bestiter, frames = train_min_radius_boundary_R4(
        degree=args.degree, k=args.k,
        n_boundary=args.n_boundary,
        region=region,
        n_iters=args.n_iters, lr=args.lr, seed=args.seed,
        polynomial_bound=1e-2,
        report_every=max(5, args.n_iters//100),
        animate=args.anim, max_frames=60,
        clip_grad_norm=5.0,
        tau=0.0
    )
    
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir,"history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["it","loss","R", "R_smooth","grad_norm","rmean","rvar","cx","cy","cp","cq"])
        writer.writeheader()
        for row in hist: writer.writerow(row)
    
    if args.anim:
        save_projection_animation(frames, outpath=os.path.join(args.outdir,"training.mp4"))
        save_radial_projection_animation(frames, outpath=os.path.join(args.outdir,"training_radial.mp4"))

    (x1B,x2B,y1B,y2B) = startB
    (X1,X2,Y1,Y2) = endB
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    axes[0].scatter(x1B, y1B, s=1, label="input boundary")
    axes[1].scatter(X1, Y1, s=1, label="mapped boundary")
    for ax in axes:
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_title("Projection (x1,y1): input")
    lastR = hist[-1]['R'] if hist else 0.0
    axes[1].set_title(f"Projection (x1,y1): mapped (R≈{lastR:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir,"snapshot.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
