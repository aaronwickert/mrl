"""
Trajectory Registration Algorithm Comparison

Compares multiple algorithms for aligning 3D trajectories:
1. Kabsch (SVD) - Optimal rigid transformation
2. Kabsch with Scaling (Umeyama) - Rigid + uniform scale
3. Horn's Quaternion Method - Quaternion-based rotation estimation
4. RANSAC + Kabsch - Robust to outliers
5. ICP (Iterative Closest Point) - No correspondence assumption
6. Point-to-Point ICP with scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from traj_extractor import (
    extract_trajectory_from_bags,
    get_rerun_entity_data,
)
from trajectory_transform import extract_trajectory_from_dataframe


@dataclass
class RegistrationResult:
    """Result of a trajectory registration algorithm."""
    name: str
    R: np.ndarray
    t: np.ndarray
    scale: float
    rmse: float
    aligned_trajectory: np.ndarray
    iterations: int = 1


# =============================================================================
# Algorithm 1: Kabsch (SVD-based, no scaling)
# =============================================================================
def kabsch_algorithm(source: np.ndarray, target: np.ndarray) -> RegistrationResult:
    """
    Kabsch algorithm using SVD for optimal rotation.
    Minimizes RMSD between corresponding points.
    """
    P = source[:, :3].astype(np.float64)
    Q = target[:, :3].astype(np.float64)

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - R @ centroid_P
    aligned = (P @ R.T) + t
    rmse = np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean())

    return RegistrationResult(
        name="Kabsch (SVD)",
        R=R, t=t, scale=1.0, rmse=rmse,
        aligned_trajectory=aligned
    )


# =============================================================================
# Algorithm 2: Umeyama (Kabsch + Scaling)
# =============================================================================
def umeyama_algorithm(source: np.ndarray, target: np.ndarray) -> RegistrationResult:
    """
    Umeyama algorithm - Kabsch with optimal uniform scaling.
    """
    P = source[:, :3].astype(np.float64)
    Q = target[:, :3].astype(np.float64)

    n = P.shape[0]
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        S[-1] = -S[-1]
        R = Vt.T @ U.T

    var_P = (P_centered ** 2).sum() / n
    scale = S.sum() / var_P

    t = centroid_Q - scale * (R @ centroid_P)
    aligned = scale * (P @ R.T) + t
    rmse = np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean())

    return RegistrationResult(
        name="Umeyama (SVD + Scale)",
        R=R, t=t, scale=scale, rmse=rmse,
        aligned_trajectory=aligned
    )


# =============================================================================
# Algorithm 3: Horn's Quaternion Method
# =============================================================================
def horn_quaternion_algorithm(source: np.ndarray, target: np.ndarray) -> RegistrationResult:
    """
    Horn's method using unit quaternions for rotation estimation.
    """
    P = source[:, :3].astype(np.float64)
    Q = target[:, :3].astype(np.float64)

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Build the 4x4 matrix M for quaternion estimation
    M = np.zeros((4, 4))
    for i in range(len(P)):
        p = P_centered[i]
        q = Q_centered[i]

        # Cross-covariance contribution
        Sxx = p[0] * q[0]
        Sxy = p[0] * q[1]
        Sxz = p[0] * q[2]
        Syx = p[1] * q[0]
        Syy = p[1] * q[1]
        Syz = p[1] * q[2]
        Szx = p[2] * q[0]
        Szy = p[2] * q[1]
        Szz = p[2] * q[2]

        N = np.array([
            [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
            [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
            [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
            [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]
        ])
        M += N

    # Eigenvector corresponding to largest eigenvalue is the optimal quaternion
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    q_opt = eigenvectors[:, -1]  # Largest eigenvalue

    # Convert quaternion to rotation matrix
    # Quaternion format: [w, x, y, z]
    w, x, y, z = q_opt
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    t = centroid_Q - R @ centroid_P
    aligned = (P @ R.T) + t
    rmse = np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean())

    return RegistrationResult(
        name="Horn (Quaternion)",
        R=R, t=t, scale=1.0, rmse=rmse,
        aligned_trajectory=aligned
    )


# =============================================================================
# Algorithm 4: RANSAC + Kabsch (Robust)
# =============================================================================
def ransac_kabsch_algorithm(
    source: np.ndarray,
    target: np.ndarray,
    n_iterations: int = 1000,
    sample_size: int = 10,
    inlier_threshold: float = 0.5
) -> RegistrationResult:
    """
    RANSAC-based robust registration using Kabsch.
    Handles outliers by iteratively finding best inlier set.
    """
    P = source[:, :3].astype(np.float64)
    Q = target[:, :3].astype(np.float64)
    n_points = len(P)

    best_inliers = None
    best_n_inliers = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)

    for _ in range(n_iterations):
        # Random sample
        indices = np.random.choice(n_points, size=min(sample_size, n_points), replace=False)
        P_sample = P[indices]
        Q_sample = Q[indices]

        # Compute transformation from sample
        centroid_P = P_sample.mean(axis=0)
        centroid_Q = Q_sample.mean(axis=0)
        P_centered = P_sample - centroid_P
        Q_centered = Q_sample - centroid_Q

        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_Q - R @ centroid_P

        # Count inliers
        aligned = (P @ R.T) + t
        distances = np.linalg.norm(aligned - Q, axis=1)
        inliers = distances < inlier_threshold
        n_inliers = inliers.sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inliers
            best_R = R
            best_t = t

    # Refine with all inliers
    if best_inliers is not None and best_n_inliers > 3:
        P_inliers = P[best_inliers]
        Q_inliers = Q[best_inliers]

        centroid_P = P_inliers.mean(axis=0)
        centroid_Q = Q_inliers.mean(axis=0)
        P_centered = P_inliers - centroid_P
        Q_centered = Q_inliers - centroid_Q

        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        best_R = R
        best_t = centroid_Q - R @ centroid_P

    aligned = (P @ best_R.T) + best_t
    rmse = np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean())

    return RegistrationResult(
        name=f"RANSAC + Kabsch ({best_n_inliers}/{n_points} inliers)",
        R=best_R, t=best_t, scale=1.0, rmse=rmse,
        aligned_trajectory=aligned,
        iterations=n_iterations
    )


# =============================================================================
# Algorithm 5: ICP (Iterative Closest Point)
# =============================================================================
def icp_algorithm(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> RegistrationResult:
    """
    Iterative Closest Point algorithm.
    Does not assume known correspondences.
    """
    P = source[:, :3].astype(np.float64).copy()
    Q = target[:, :3].astype(np.float64)

    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = float('inf')

    tree = KDTree(Q)
    iterations_used = 0

    for i in range(max_iterations):
        iterations_used = i + 1

        # Find closest points
        distances, indices = tree.query(P)
        Q_matched = Q[indices]

        # Compute transformation
        centroid_P = P.mean(axis=0)
        centroid_Q = Q_matched.mean(axis=0)
        P_centered = P - centroid_P
        Q_centered = Q_matched - centroid_Q

        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_Q - R @ centroid_P

        # Update total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        # Apply transformation
        P = (P @ R.T) + t

        # Check convergence
        mean_error = distances.mean()
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    rmse = np.sqrt(((P - Q[tree.query(P)[1]]) ** 2).sum(axis=1).mean())

    return RegistrationResult(
        name=f"ICP ({iterations_used} iters)",
        R=R_total, t=t_total, scale=1.0, rmse=rmse,
        aligned_trajectory=P,
        iterations=iterations_used
    )


# =============================================================================
# Algorithm 6: ICP with Scaling
# =============================================================================
def icp_with_scaling_algorithm(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> RegistrationResult:
    """
    ICP with uniform scaling estimation.
    """
    P = source[:, :3].astype(np.float64).copy()
    Q = target[:, :3].astype(np.float64)
    P_original = P.copy()

    R_total = np.eye(3)
    t_total = np.zeros(3)
    scale_total = 1.0
    prev_error = float('inf')

    tree = KDTree(Q)
    iterations_used = 0

    for i in range(max_iterations):
        iterations_used = i + 1

        # Find closest points
        distances, indices = tree.query(P)
        Q_matched = Q[indices]

        # Compute transformation with scaling (Umeyama)
        n = len(P)
        centroid_P = P.mean(axis=0)
        centroid_Q = Q_matched.mean(axis=0)
        P_centered = P - centroid_P
        Q_centered = Q_matched - centroid_Q

        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            S[-1] = -S[-1]
            R = Vt.T @ U.T

        var_P = (P_centered ** 2).sum() / n
        scale = S.sum() / var_P if var_P > 1e-10 else 1.0
        scale = np.clip(scale, 0.5, 2.0)  # Prevent extreme scaling

        t = centroid_Q - scale * (R @ centroid_P)

        # Update totals
        R_total = R @ R_total
        t_total = scale * (R @ t_total) + t
        scale_total *= scale

        # Apply transformation
        P = scale * (P @ R.T) + t

        # Check convergence
        mean_error = distances.mean()
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    aligned = scale_total * (P_original @ R_total.T) + t_total
    rmse = np.sqrt(((aligned - Q[tree.query(aligned)[1]]) ** 2).sum(axis=1).mean())

    return RegistrationResult(
        name=f"ICP + Scale ({iterations_used} iters)",
        R=R_total, t=t_total, scale=scale_total, rmse=rmse,
        aligned_trajectory=aligned,
        iterations=iterations_used
    )


# =============================================================================
# Visualization
# =============================================================================
def plot_all_results(
    source: np.ndarray,
    target: np.ndarray,
    results: List[RegistrationResult],
    source_label: str = "Source",
    target_label: str = "Target"
) -> None:
    """
    Create a subplot for each registration algorithm result.
    """
    n_results = len(results)
    n_cols = 3
    n_rows = (n_results + n_cols - 1) // n_cols + 1  # +1 for original

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    # Plot original (unaligned)
    ax0 = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    ax0.plot(source[:, 0], source[:, 1], source[:, 2],
             'b-', linewidth=1, label=source_label, alpha=0.7)
    ax0.plot(target[:, 0], target[:, 1], target[:, 2],
             'r-', linewidth=1, label=target_label, alpha=0.7)
    ax0.set_title("Original (Unaligned)", fontsize=12, fontweight='bold')
    ax0.legend(fontsize=8)
    _set_equal_axes(ax0, source, target)

    # Plot each algorithm result
    for i, result in enumerate(results):
        ax = fig.add_subplot(n_rows, n_cols, i + 2, projection='3d')
        ax.plot(result.aligned_trajectory[:, 0],
                result.aligned_trajectory[:, 1],
                result.aligned_trajectory[:, 2],
                'b-', linewidth=1, label=f"{source_label} (aligned)", alpha=0.7)
        ax.plot(target[:, 0], target[:, 1], target[:, 2],
                'r-', linewidth=1, label=target_label, alpha=0.7)

        title = f"{result.name}\nRMSE: {result.rmse:.4f}m"
        if result.scale != 1.0:
            title += f", Scale: {result.scale:.3f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        _set_equal_axes(ax, result.aligned_trajectory, target)

    plt.tight_layout()
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/registration_comparison_3d.svg',
                format='svg', bbox_inches='tight')
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/registration_comparison_3d.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # Create 2D comparison plot (XY projection)
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Original
    axes[0].plot(source[:, 0], source[:, 1], 'b-', linewidth=1, label=source_label, alpha=0.7)
    axes[0].plot(target[:, 0], target[:, 1], 'r-', linewidth=1, label=target_label, alpha=0.7)
    axes[0].set_title("Original (Unaligned)", fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    for i, result in enumerate(results[:5]):  # First 5 results
        ax = axes[i + 1]
        ax.plot(result.aligned_trajectory[:, 0], result.aligned_trajectory[:, 1],
                'b-', linewidth=1, label=f"{source_label} (aligned)", alpha=0.7)
        ax.plot(target[:, 0], target[:, 1],
                'r-', linewidth=1, label=target_label, alpha=0.7)
        title = f"{result.name}\nRMSE: {result.rmse:.4f}m"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/registration_comparison_2d.svg',
                format='svg', bbox_inches='tight')
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/registration_comparison_2d.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # Error comparison bar chart
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = [r.name.split('\n')[0] for r in results]
    rmses = [r.rmse for r in results]
    scales = [r.scale for r in results]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    bars = ax1.bar(range(len(results)), rmses, color=colors)
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('RMSE (m)', fontsize=11)
    ax1.set_title('RMSE Comparison by Algorithm', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{rmse:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.bar(range(len(results)), scales, color=colors)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Scale Factor', fontsize=11)
    ax2.set_title('Scale Factor by Algorithm', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='No scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/registration_comparison_metrics.svg',
                format='svg', bbox_inches='tight')
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/registration_comparison_metrics.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def _set_equal_axes(ax, *trajectories):
    """Set equal axis scaling for 3D plot."""
    all_points = np.vstack([t[:, :3] for t in trajectories])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def resample_trajectory(traj: np.ndarray, num_points: int) -> np.ndarray:
    """Resample trajectory to specified number of points."""
    n = len(traj)
    if n == num_points:
        return traj[:, :3].copy()

    old_indices = np.linspace(0, 1, n)
    new_indices = np.linspace(0, 1, num_points)

    resampled = np.zeros((num_points, 3))
    for i in range(3):
        resampled[:, i] = np.interp(new_indices, old_indices, traj[:, i])

    return resampled


def print_results_table(results: List[RegistrationResult]) -> None:
    """Print a formatted table of results."""
    print("\n" + "=" * 80)
    print("REGISTRATION RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Algorithm':<35} {'RMSE (m)':<12} {'Scale':<10} {'Iters':<8}")
    print("-" * 80)

    # Sort by RMSE
    sorted_results = sorted(results, key=lambda x: x.rmse)

    for r in sorted_results:
        name = r.name[:34]
        print(f"{name:<35} {r.rmse:<12.6f} {r.scale:<10.4f} {r.iterations:<8}")

    print("-" * 80)
    best = sorted_results[0]
    print(f"Best algorithm: {best.name} with RMSE = {best.rmse:.6f}m")
    print("=" * 80)


def main():
    # Paths
    rrd_path = "/home/aaron/PycharmProjects/PythonProject/data.rrd"
    mocap_bag_path = "/home/aaron/uni/thesis/data/seq4/seq4_mocap"
    mocap_topic = "/qualisys/TUDA_swift/pose"

    print("=" * 60)
    print("Trajectory Registration Algorithm Comparison")
    print("=" * 60)

    # Extract trajectories
    print("\n[1] Extracting trajectories...")

    # RRD trajectory
    result = get_rerun_entity_data(rrd_path, "/world/trajectory", timeline="data_time")
    traj_rrd = extract_trajectory_from_dataframe(result["data"])
    print(f"    RRD trajectory: {traj_rrd.shape}")

    # MoCap trajectory
    traj_mocap = extract_trajectory_from_bags(mocap_bag_path, mocap_topic)
    traj_mocap_xyz = traj_mocap[:, :3]
    print(f"    MoCap trajectory: {traj_mocap_xyz.shape}")

    # Resample to match lengths
    print("\n[2] Resampling trajectories...")
    num_points = min(len(traj_rrd), len(traj_mocap_xyz))
    source = resample_trajectory(traj_rrd, num_points)
    target = resample_trajectory(traj_mocap_xyz, num_points)
    print(f"    Resampled to {num_points} points each")

    # Run all algorithms
    print("\n[3] Running registration algorithms...")
    results = []

    print("    - Kabsch (SVD)...")
    results.append(kabsch_algorithm(source, target))

    print("    - Umeyama (SVD + Scale)...")
    results.append(umeyama_algorithm(source, target))

    print("    - Horn (Quaternion)...")
    results.append(horn_quaternion_algorithm(source, target))

    print("    - RANSAC + Kabsch...")
    results.append(ransac_kabsch_algorithm(source, target, n_iterations=1000, inlier_threshold=0.5))

    print("    - ICP...")
    results.append(icp_algorithm(source, target, max_iterations=100))

    print("    - ICP + Scale...")
    results.append(icp_with_scaling_algorithm(source, target, max_iterations=100))

    # Print results table
    print_results_table(results)

    # Visualize
    print("\n[4] Generating visualizations...")
    plot_all_results(source, target, results,
                     source_label="RRD (SLAM)",
                     target_label="MoCap (GT)")

    print("\nPlots saved to thesis_resources/:")
    print("  - registration_comparison_3d.svg / .png")
    print("  - registration_comparison_2d.svg / .png")
    print("  - registration_comparison_metrics.svg / .png")


if __name__ == "__main__":
    main()
