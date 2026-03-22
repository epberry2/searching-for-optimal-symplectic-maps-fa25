import IMLR4 as sym
import numpy as np
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as joptimizers
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, csv, time

def main():
    d = 2 # polynomial degree is d * 2, will be a d+1 x d+1 coeff matrix
    k = 40 # number of Henon maps in composition

    radii = (1,np.sqrt(4),1, np.sqrt(4))
    #region = sym.Ellipsoid4D(radii=radii)
    region = sym.PolyDisk4D(a=1.0, b=6.0)
    #region = sym.LagrangianTorus4D(a=1.0, b=6.0)

    loaded_params = np.load("output/final_params.npy")
        # generate and save snapshot of final map
    # note that this is a different set of boundary points than used in training
    x1B, x2B, y1B, y2B = region.boundary_points(k=int(1e7), seed=100)
    X1B, X2B, Y1B, Y2B = sym.henon_comp_forward_jax(loaded_params, d, k, x1B, x2B, y1B, y2B)
    fig, axes = plt.subplots(1,2, figsize=(9,4.5))
    axes[0].scatter(X1B, Y1B, s=1)
    axes[1].scatter(X2B, Y2B, s=1)
    for ax,title in zip(axes, ["Projection (x1,y1)", "Projection (x2,y2)"]):
        ax.set_aspect("equal")
        ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    fig.savefig(os.path.join("output_r4/visualizations","polydisk.png"), dpi=150)

    print("Saved snapshot of final map to output_r4/visualizations/polydisk.png")

    # save radial projection of final map
    r1 = X1B**2 + Y1B**2
    r2 = X2B**2 + Y2B**2
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.scatter(r1, r2, s=1)
    ax.set_aspect("equal")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Radial Projection")
    fig.savefig(os.path.join("output_r4/visualizations","radial_polydisk.png"), dpi=150)

    print("Saved radial projection snapshot to output_r4/visualizations/radial_polydisk.png")


    final_R = float(jnp.max(X1B**2 + X2B**2 + Y1B**2 + Y2B**2))
    print(f"Final max radius on boundary: {final_R:.6f}")

    # Create visualization directory
    os.makedirs("output_r4/visualizations", exist_ok=True)

    # Generate all visualizations
    visualize_3d_with_color(X1B, X2B, Y1B, Y2B, 
        outpath=os.path.join("output_r4/visualizations", "polydisk_3d_projection.png"))
    visualize_density_heatmap(X1B, X2B, Y1B, Y2B,
        outpath=os.path.join("output_r4/visualizations", "polydisk_density_heatmap.png"))




def visualize_3d_with_color(X1, X2, Y1, Y2, title="3D Projection", outpath=None):
    """3D scatter: (x1, x2, y1) colored by radius."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    radius = np.sqrt(X1**2 + X2**2 + Y1**2 + Y2**2)
    scatter = ax.scatter(X1, X2, Y1, c=radius, cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y1')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Radius')
    
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return fig

def visualize_density_heatmap(X1, X2, Y1, Y2, outpath=None):
    """2D density heatmaps for all coordinate pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    pairs = [
        (X1, X2, 'x1', 'x2'),
        (X1, Y1, 'x1', 'y1'),
        (X1, Y2, 'x1', 'y2'),
        (X2, Y1, 'x2', 'y1'),
        (X2, Y2, 'x2', 'y2'),
        (Y1, Y2, 'y1', 'y2'),
    ]
    
    axes = axes.flatten()
    for ax, (x, y, xlabel, ylabel) in zip(axes, pairs):
        h = ax.hist2d(x, y, bins=100, cmap='hot')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        plt.colorbar(h[3], ax=ax, label='Count')
    
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return fig

def load_and_evaluate_params(params_path, degree, k, region, n_points=int(5e5)):
    """Load saved parameters and evaluate them on new boundary points.
    
    Args:
        params_path: Path to the saved .npy file
        degree: Polynomial degree
        k: Number of Henon maps
        region: Shape4D region object for sampling boundary points
        n_points: Number of boundary points to evaluate
    
    Returns:
        X1B, X2B, Y1B, Y2B: Mapped coordinates
    """
    final_params = np.load(params_path)
    print(f"Loaded parameters from {params_path}")
    
    x1B, x2B, y1B, y2B = region.boundary_points(k=n_points, seed=100)
    X1B, X2B, Y1B, Y2B = sym.henon_comp_forward_jax(final_params, degree, k, x1B, x2B, y1B, y2B)
    
    return X1B, X2B, Y1B, Y2B


if __name__ == '__main__':
    main()

