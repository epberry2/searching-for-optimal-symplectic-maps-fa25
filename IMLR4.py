import numpy as np
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as joptimizers
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, csv, time


# Loss function for JAX
def loss_max_radius_boundary_R4_jax(params, x1B, x2B, y1B, y2B, degree, k, w_center=1e-3, w_reg=1e-7, tau=None):
    # forward pass through HenonComp
    X1, X2, Y1, Y2 = henon_comp_forward_jax(params, degree, k, x1B, x2B, y1B, y2B)
    r = jnp.sqrt(X1**2 + X2**2 + Y1**2 + Y2**2)
    R = jnp.max(r) ** 2
    if tau is None or tau == 0:
        Lmax = jnp.max(r)
    else:
        # Use LogSumExp to approximate max
        t = tau * r
        tmax = jnp.max(t)
        s = jnp.sum(jnp.exp(t - tmax))
        Lmax = (tmax + jnp.log(s)) / tau
    # centering and regularization
    cx = jnp.mean(X1); cy = jnp.mean(X2); cp = jnp.mean(Y1); cq = jnp.mean(Y2)
    reg = jnp.sum(params**2)
    L = Lmax + w_center * (cx**2 + cy**2 + cp**2 + cq**2) + w_reg * reg
    return L, R


def polyval2d(y1, y2, coeffs):
    # evaluate 2D polynomial with given coeffs at (y1, y2)
    D = coeffs.shape[0]
    val = jnp.zeros_like(y1)
    for i in range(D):
        for j in range(D):
            val += coeffs[i, j] * (y1 ** (D - 1 - i)) * (y2 ** (D - 1 - j))
    return val

# One Henon map application
def henon_map_apply_jax(x1, x2, y1, y2, coeffs, const):
    D = coeffs.shape[0]
    dV_dy1 = jnp.zeros_like(y1)
    dV_dy2 = jnp.zeros_like(y2)

    # create gradients of V w.r.t. y1, y2
    for i in range(D):
        for j in range(D):
            c = coeffs[i, j]
            p1 = D - 1 - i
            p2 = D - 1 - j
            if p1 > 0:
                dV_dy1 = dV_dy1 + c * p1 * (y1 ** (p1 - 1)) * (y2 ** p2)
            if p2 > 0:
                dV_dy2 = dV_dy2 + c * p2 * (y1 ** p1) * (y2 ** (p2 - 1))

    return (
        y1 + const[0],
        y2 + const[1],
        -x1 + dV_dy1,
        -x2 + dV_dy2,
    )

# Composition of Henon maps
def henon_comp_forward_jax(params, degree, k, x1, x2, y1, y2):
    D = degree + 1
    off = 0
    for i in range(k - 1, -1, -1):
        coeffs = params[off : off + D*D].reshape(D, D) # parameters for polynomials
        off += D*D
        const = params[off : off + 2] # parameters for constants
        off += 2
        x1, x2, y1, y2 = henon_map_apply_jax(x1, x2, y1, y2, coeffs, const)
    return x1, x2, y1, y2

class Shape4D:
    def volume(self): return 1.0
    def boundary_points(self, k=1, seed=0): raise NotImplementedError

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
    def __init__(self, a=1.0, b=2.0):
        self.a = float(a)
        self.b = float(b)
    def boundary_points(self, k=2048, seed=0):
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0, 2*np.pi, k)
        t2 = rng.uniform(0, 2*np.pi, k)
        R1 = np.sqrt(self.a / np.pi)
        R2 = np.sqrt(self.b / np.pi)
        x1 = R1 * np.cos(t1)
        y1 = R1 * np.sin(t1)
        x2 = R2 * np.cos(t2)
        y2 = R2 * np.sin(t2)
        return x1, x2, y1, y2
    
class PolyDisk4D(Shape4D):
    def __init__(self, a=1.0, b=2.5):
        self.a = float(a)
        self.b = float(b)

    def boundary_points(self, k=2048, seed=0):
        rng = np.random.default_rng(seed)

        length = np.sqrt(rng.uniform(0, 1, (k, 2)))
        angle  = np.pi * rng.uniform(0, 2, (k, 2))

        half = k // 2
        length[:half, 0] = 1.0
        length[half:, 1] = 1.0

        q = length * np.cos(angle)
        p = length * np.sin(angle)

        R1 = np.sqrt(self.a / np.pi)
        R2 = np.sqrt(self.b / np.pi)

        z1 = R1 * np.column_stack([q[:, 0], p[:, 0]])
        z2 = R2 * np.column_stack([q[:, 1], p[:, 1]]) 

        z = np.column_stack([z1[:, 0], z1[:, 1], z2[:, 0], z2[:, 1]]).astype(np.float32)

        x1 = z[:, 0]
        x2 = z[:, 2]
        y1 = z[:, 1]
        y2 = z[:, 3]

        return x1, x2, y1, y2

# training function to minimize max radius on boundary of 4D shape
def train_min_radius_boundary_R4(
    degree=3, k=20,
    n_boundary=5000,
    region=Ellipsoid4D(radii=(1,4,1,4)),
    n_iters=200, lr=2e-3, seed=11,
    polynomial_bound=5e-3,
    w_center=1e-3, w_reg=1e-7, report_every=25,
    optimizer='adam',
    minibatch_size=None,
    lr_backoff_factor=0.5,
    min_lr=1e-12,
    max_step_retries=5,
    animate=False, max_frames=100,
    tau=30.0,
    initial_params=None
):
    timestart = time.time()
    D = degree + 1
    num_params = k * D * D + 2 * k
    rng = np.random.default_rng(seed)
    
    # Use provided params or initialize new ones
    if initial_params is not None:
        theta = np.asarray(initial_params, dtype=np.float32)
        print("Initializing from provided parameters")
    else:
        theta = rng.normal(scale=polynomial_bound, size=num_params) # initial params

    history = []
    frames = []

    x1B_all, x2B_all, y1B_all, y2B_all = region.boundary_points(k=n_boundary, seed=seed+1) # sample boundary points

    current_lr = lr

    # initialize optimizer
    if optimizer == 'adam':
        opt_init, opt_update, get_params = joptimizers.adam(current_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    opt_state = opt_init(theta)

    # gradient clipping for stability
    def clip_grads(grads, max_norm=10.0):
        leaves = jtu.tree_leaves(grads)
        g2 = sum([jnp.sum(g**2) for g in leaves])
        g_norm = jnp.sqrt(g2)
        factor = jnp.minimum(1.0, max_norm / (g_norm + 1e-8))
        return jtu.tree_map(lambda g: g * factor, grads)

    @jax.jit
    def compute_loss_and_grads(params, x1B, x2B, y1B, y2B):
        (loss, R), grads = jax.value_and_grad(loss_max_radius_boundary_R4_jax, has_aux=True)(
            params, x1B, x2B, y1B, y2B,
            degree, k,
            w_center=w_center,
            w_reg=w_reg,
            tau = tau
        )
        grads = clip_grads(grads, max_norm=10.0)
        return loss, R, grads

    # training loop
    for it in range(n_iters):
        if minibatch_size is not None:
            indices = rng.choice(n_boundary, size=minibatch_size, replace=False)
            x1B_batch = x1B_all[indices]
            x2B_batch = x2B_all[indices]
            y1B_batch = y1B_all[indices]
            y2B_batch = y2B_all[indices]
        else:
            x1B_batch = x1B_all
            x2B_batch = x2B_all
            y1B_batch = y1B_all
            y2B_batch = y2B_all
        
        # attempt the step and, on divergence, retry on the same iteration
        params = get_params(opt_state)
        old_opt_state = opt_state
        attempt = 0
        accepted = False
        while attempt <= max_step_retries:
            # compute minibatch loss & grads
            loss_mb, R_mb, grads = compute_loss_and_grads(params, x1B_batch, x2B_batch, y1B_batch, y2B_batch)
            # propose optimizer state by applying the update with current opt_update
            opt_state_candidate = opt_update(0, grads, opt_state)
            params_candidate = get_params(opt_state_candidate)

            # evaluate full-batch loss on the proposed params for safety
            full_loss, full_R, _ = compute_loss_and_grads(params_candidate, x1B_all, x2B_all, y1B_all, y2B_all)

            if jnp.isfinite(full_loss) and (full_loss <= 1e12):
                # accept the candidate
                opt_state = opt_state_candidate
                params = params_candidate
                # record accepted loss/R for downstream reporting
                loss = full_loss
                R = full_R
                history.append(dict(it=it, loss=full_loss, R=full_R))
                accepted = True
                break

            # otherwise rollback and back off
            attempt += 1
            print(f"Iter {it} attempt {attempt}: proposed step diverged (full_loss={float(full_loss)}). Rolling back.")
            # restore previous optimizer state
            opt_state = old_opt_state
            # check retry limits
            if attempt > max_step_retries or float(current_lr) * float(lr_backoff_factor) < float(min_lr):
                print("Max retries reached or LR below min; stopping training.")
                accepted = False
                break
            # reduce LR and reinitialize optimizer state from current params
            current_lr = float(current_lr) * float(lr_backoff_factor)
            print(f"Backing off LR to {current_lr:.3e} and reinitializing optimizer state.")
            if optimizer == 'adam':
                opt_init, opt_update, get_params = joptimizers.adam(current_lr)
            opt_state = opt_init(params)

        if not accepted:
            print("Step failed after retries; stopping training.")
            break
        if animate and (it % (n_iters // max_frames) == 0 or it == n_iters - 1):
            
            params = get_params(opt_state)
            X1B, X2B, Y1B, Y2B = henon_comp_forward_jax(params, degree, k, x1B_all, x2B_all, y1B_all, y2B_all)
            frames.append((X1B, Y1B, X2B, Y2B, R, it))
            
        if (it + 1) % report_every == 0 or it == 0:
            params = get_params(opt_state)
            print(f"Iter {it+1}, Loss: {float(loss):.6f}, R: {float(R):.6f}")

    timeend = time.time()
    print(f"Training completed in {timeend - timestart:.2f} seconds.")
    final_params = get_params(opt_state)
    return final_params, history, frames

# save 2d projection animation
def save_projection_animation(frames, outpath="r4_training.gif", writer="pillow"):
    if not frames:
        print("No frames to animate."); return
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    scat1 = axes[0].scatter([], [], s=1)
    scat2 = axes[1].scatter([], [], s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-4,4); ax.set_ylim(-4,4)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    def update(frame):
        X1,Y1,X2,Y2,R, it = frame
        scat1.set_offsets(np.column_stack([X1, Y1]))
        scat2.set_offsets(np.column_stack([X2, Y2]))
        axes[0].set_title(f"(x1,y1)  R≈{R:.3f}  iter={it}")
        axes[1].set_title(f"(x2,y2)  R≈{R:.3f}  iter={it}")
        return scat1, scat2
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer=writer)
    plt.close(fig)
    print(f"Saved animation to {outpath}")

# save radial projection animation
def save_radial_projection_animation(frames, outpath="r4_radial_training.gif", writer="pillow"):
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
    for X1, Y1, X2, Y2, _, _ in frames:
        r1 = (X1**2 + Y1**2)
        r2 = (X2**2 + Y2**2)
        if r1.size:
            all_r1_min = min(all_r1_min, float(r1.min())); all_r1_max = max(all_r1_max, float(r1.max()))
        if r2.size:
            all_r2_min = min(all_r2_min, float(r2.min())); all_r2_max = max(all_r2_max, float(r2.max()))

    pad1 = 0.05 * (all_r1_max - all_r1_min) if all_r1_max>all_r1_min else 0.1
    pad2 = 0.05 * (all_r2_max - all_r2_min) if all_r2_max>all_r2_min else 0.1

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    scat = ax.scatter([], [], s=1)
    ax.set_aspect('equal')
    if all_r1_max > 10 or all_r2_max > 10:
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
    else:
        ax.set_xlim(all_r1_min - pad1, all_r1_max + pad1)
        ax.set_ylim(all_r2_min - pad2, all_r2_max + pad2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Radial projection (x1^2+y1^2 vs x2^2+y2^2)")

    def update(frame):
        X1, Y1, X2, Y2, R, it = frame
        r1 = (X1**2 + Y1**2)
        r2 = (X2**2 + Y2**2)
        coords = np.column_stack([r1, r2])
        scat.set_offsets(coords)
        ax.set_title(f"Radial projection  R≈{R:.3f}  iter={it}")
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer=writer)
    plt.close(fig)
    print(f"Saved radial projection animation to {outpath}")

def main():
    d = 2 # polynomial degree is d * 2, will be a d+1 x d+1 coeff matrix
    k = 40 # number of Henon maps in composition

    radii = (1,np.sqrt(4),1, np.sqrt(4))
    #region = Ellipsoid4D(radii=radii)
    #region = PolyDisk4D(a=1.0, b=6.0)
    region = LagrangianTorus4D(a=1.0, b=6.0)

    loaded_params = np.load("output_r4/final_params.npy")

    print("Starting Training")
    final_params, history, frames = train_min_radius_boundary_R4(
        degree=d, k=k,
        n_boundary=10000,
        region=region,
        n_iters=1000, lr=1e-4, seed=54,
        polynomial_bound=0,
        w_center=1e-3, w_reg=1e-7, report_every=1000,
        optimizer='adam',
        minibatch_size=None,
        animate=False, max_frames=1,
        tau = 50,
        initial_params=loaded_params
    )

    # Save final parameters as binary
    os.makedirs("output_r4", exist_ok=True)
    np.save(os.path.join("output_r4", "final_params.npy"), final_params)
    print(f"Saved final parameters to output_r4/final_params.npy")

    # write history to CSV
    os.makedirs("output_r4", exist_ok=True)
    with open(os.path.join("output_r4","history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["it","loss", "R"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    # save animations
    if frames:
        save_projection_animation(frames, outpath=os.path.join("output_r4","training_animation.gif"), writer="pillow")
        save_radial_projection_animation(frames, outpath=os.path.join("output_r4","training_radial_animation.gif"), writer="pillow")

    # save final parameters
    with open(os.path.join("output_r4","params.txt"), "w", newline="") as f:
        off = 0
        for i in range(k - 1, -1, -1):
            coeffs = final_params[off : off + (d+1)*(d+1)].reshape(d+1, d+1)
            f.write(f"# Henon map {k - i} coefficients:\n")
            for row in coeffs:
                f.write("# " + ", ".join(f"{c:.6e}" for c in row) + "\n")
            off += (d+1)*(d+1)
            const = final_params[off : off + 2]
            f.write(f"# Henon map {k - i} constants:\n{const[0]:.6e}, {const[1]:.6e}\n")
            off += 2

    # generate and save snapshot of final map
    # note that this is a different set of boundary points than used in training
    x1B, x2B, y1B, y2B = region.boundary_points(k=int(1e5), seed=100)
    X1B, X2B, Y1B, Y2B = henon_comp_forward_jax(final_params, d, k, x1B, x2B, y1B, y2B)
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    axes[0].scatter(X1B, Y1B, s=1)
    axes[1].scatter(X2B, Y2B, s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    fig.savefig(os.path.join("output_r4","snapshot.png"), dpi=150)

    print("Saved snapshot of final map to output_r4/snapshot.png")

    # save radial projection of final map
    r1 = X1B**2 + Y1B**2
    r2 = X2B**2 + Y2B**2
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.scatter(r1, r2, s=1)
    ax.set_aspect("equal")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Radial Projection")
    fig.savefig(os.path.join("output_r4","radial_snapshot.png"), dpi=150)

    print("Saved radial projection snapshot to output_r4/radial_snapshot.png")


    final_R = float(jnp.max(X1B**2 + X2B**2 + Y1B**2 + Y2B**2))
    print(f"Final max radius on boundary: {final_R:.6f}")

    print("Computing Symplectic Error")
    # compute Jacobians and symplectic errors

    # wrapper mapping a 4-vector -> 4-vector using the composition
    def henon_comp_point(z, params, degree, k):
        x1, x2, y1, y2 = z
        # henon_comp_forward_jax expects arrays; call with length-1 arrays and squeeze
        x1p, x2p, y1p, y2p = henon_comp_forward_jax(
            params, degree, k,
            jnp.array([x1]), jnp.array([x2]), jnp.array([y1]), jnp.array([y2])
        )
        return jnp.stack([x1p[0], x2p[0], y1p[0], y2p[0]])

    # Jacobian w.r.t. the input point z (shape (4,4))
    jac_fn = jax.jacfwd(henon_comp_point, argnums=0)  # or jax.jacrev

    params = jnp.asarray(final_params, dtype=jnp.float32)

    # Symplectic matrix S for ordering (x1,x2,y1,y2)
    I2 = jnp.eye(2)
    Z2 = jnp.zeros((2,2))
    S = jnp.block([[Z2, I2], [-I2, Z2]])

    # batch of boundary points to evaluate Jacobians on
    points = jnp.stack([x1B, x2B, y1B, y2B], axis=1)  
    batched_jac = jax.vmap(lambda z: jac_fn(z, params, d, k))(points)  # shape (N,4,4)

    # compute worst-case symplectic error across points
    errs = jax.vmap(lambda J: jnp.max(jnp.abs(J.T @ S @ J - S)))(batched_jac)
    print("max symplectic error over batch:", float(jnp.max(errs)))
    print("avg symplectic error over batch:", float(jnp.mean(errs)))
    

if __name__ == '__main__':
    main()