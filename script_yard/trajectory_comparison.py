import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional

from traj_extractor import (
    extract_trajectory_from_bags,
    get_rerun_entity_data,
)
from trajectory_transform import (
    compute_trajectory_transformation,
    apply_transformation,
    extract_trajectory_from_dataframe,
)


def extract_trajectory_from_rrd(
    rrd_path: str,
    entity_path: str = "/world/trajectory",
    timeline: str = "data_time"
) -> np.ndarray:
    """
    Extract trajectory data from a Rerun .rrd file.

    Parameters
    ----------
    rrd_path : str
        Path to the .rrd file
    entity_path : str
        Entity path in Rerun (default: "/world/trajectory")
    timeline : str
        Timeline to use for indexing

    Returns
    -------
    np.ndarray
        Trajectory as (N, 3) array of xyz coordinates
    """
    result = get_rerun_entity_data(rrd_path, entity_path, timeline=timeline)
    df = result["data"]
    trajectory = extract_trajectory_from_dataframe(df)
    return trajectory


def visualize_trajectory_alignment(
    traj_source: np.ndarray,
    traj_target: np.ndarray,
    traj_aligned: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float,
    rmse: float,
    source_label: str = "Source (RRD)",
    target_label: str = "Target (MoCap)",
) -> None:
    """
    Visualize trajectory alignment with before/after comparison.

    Parameters
    ----------
    traj_source : np.ndarray
        Original source trajectory (N, 3)
    traj_target : np.ndarray
        Target trajectory (M, 3)
    traj_aligned : np.ndarray
        Aligned source trajectory (N, 3)
    R : np.ndarray
        Rotation matrix (3, 3)
    t : np.ndarray
        Translation vector (3,)
    scale : float
        Scale factor applied
    rmse : float
        Root mean square error after alignment
    source_label : str
        Label for source trajectory
    target_label : str
        Label for target trajectory
    """
    fig = plt.figure(figsize=(16, 10))

    # Before alignment
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj_source[:, 0], traj_source[:, 1], traj_source[:, 2],
             'b-', linewidth=1.5, label=source_label, alpha=0.8)
    ax1.plot(traj_target[:, 0], traj_target[:, 1], traj_target[:, 2],
             'r-', linewidth=1.5, label=target_label, alpha=0.8)
    ax1.set_title("Before Alignment")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    _set_equal_axes(ax1, traj_source, traj_target)

    # After alignment
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(traj_aligned[:, 0], traj_aligned[:, 1], traj_aligned[:, 2],
             'b-', linewidth=1.5, label=f"{source_label} (aligned)", alpha=0.8)
    ax2.plot(traj_target[:, 0], traj_target[:, 1], traj_target[:, 2],
             'r-', linewidth=1.5, label=target_label, alpha=0.8)
    ax2.set_title(f"After Alignment (RMSE: {rmse:.4f} m)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()
    _set_equal_axes(ax2, traj_aligned, traj_target)

    # XY projection (top-down view)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(traj_aligned[:, 0], traj_aligned[:, 1],
             'b-', linewidth=1.5, label=f"{source_label} (aligned)", alpha=0.8)
    ax3.plot(traj_target[:, 0], traj_target[:, 1],
             'r-', linewidth=1.5, label=target_label, alpha=0.8)
    ax3.set_title("XY Projection (Top-Down)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Error analysis - point-wise distances (if same length)
    ax4 = fig.add_subplot(2, 2, 4)
    min_len = min(len(traj_aligned), len(traj_target))
    if min_len > 0:
        distances = np.linalg.norm(
            traj_aligned[:min_len] - traj_target[:min_len], axis=1
        )
        ax4.plot(distances, 'g-', linewidth=1)
        ax4.axhline(y=rmse, color='r', linestyle='--', label=f'RMSE: {rmse:.4f}')
        ax4.axhline(y=distances.mean(), color='orange', linestyle='--',
                    label=f'Mean: {distances.mean():.4f}')
        ax4.set_title("Point-wise Distance Error")
        ax4.set_xlabel("Point Index")
        ax4.set_ylabel("Distance (m)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Print statistics
        print(f"\nAlignment Statistics:")
        print(f"  RMSE: {rmse:.6f} m")
        print(f"  Mean Error: {distances.mean():.6f} m")
        print(f"  Max Error: {distances.max():.6f} m")
        print(f"  Min Error: {distances.min():.6f} m")
        print(f"  Std Dev: {distances.std():.6f} m")
    else:
        ax4.text(0.5, 0.5, "Different trajectory lengths\nCannot compute point-wise error",
                 ha='center', va='center', transform=ax4.transAxes)

    # Print transformation parameters
    print(f"\nTransformation Parameters:")
    print(f"  Scale: {scale:.6f}")
    print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  Rotation Matrix:")
    print(f"    [{R[0, 0]:8.4f}, {R[0, 1]:8.4f}, {R[0, 2]:8.4f}]")
    print(f"    [{R[1, 0]:8.4f}, {R[1, 1]:8.4f}, {R[1, 2]:8.4f}]")
    print(f"    [{R[2, 0]:8.4f}, {R[2, 1]:8.4f}, {R[2, 2]:8.4f}]")

    # Convert rotation to axis-angle for interpretability
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    print(f"  Rotation Angle: {np.degrees(angle):.2f} degrees")

    plt.tight_layout()
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
    """
    Resample trajectory to have a specific number of points using linear interpolation.

    Parameters
    ----------
    traj : np.ndarray
        Input trajectory (N, 3+)
    num_points : int
        Desired number of points

    Returns
    -------
    np.ndarray
        Resampled trajectory (num_points, 3)
    """
    n = len(traj)
    if n == num_points:
        return traj[:, :3].copy()

    old_indices = np.linspace(0, 1, n)
    new_indices = np.linspace(0, 1, num_points)

    resampled = np.zeros((num_points, 3))
    for i in range(3):
        resampled[:, i] = np.interp(new_indices, old_indices, traj[:, i])

    return resampled


def main():
    # Paths
    rrd_path = "/home/aaron/PycharmProjects/PythonProject/data.rrd"
    mocap_bag_path = "/home/aaron/uni/thesis/data/seq4/seq4_mocap"
    mocap_topic = "/qualisys/TUDA_swift/pose"

    print("=" * 60)
    print("Trajectory Comparison: RRD vs MoCap")
    print("=" * 60)

    # Extract RRD trajectory
    print("\n[1] Extracting trajectory from RRD file...")
    traj_rrd = extract_trajectory_from_rrd(rrd_path)
    print(f"    RRD trajectory shape: {traj_rrd.shape}")

    # Extract MoCap trajectory
    print("\n[2] Extracting trajectory from MoCap bags...")
    traj_mocap = extract_trajectory_from_bags(mocap_bag_path, mocap_topic)
    # Extract only xyz (first 3 columns)
    traj_mocap_xyz = traj_mocap[:, :3]
    print(f"    MoCap trajectory shape: {traj_mocap_xyz.shape}")

    # Resample to match lengths for proper alignment
    print("\n[3] Resampling trajectories to match lengths...")
    num_points = min(len(traj_rrd), len(traj_mocap_xyz))
    traj_rrd_resampled = resample_trajectory(traj_rrd, num_points)
    traj_mocap_resampled = resample_trajectory(traj_mocap_xyz, num_points)
    print(f"    Resampled to {num_points} points each")

    # Compute transformation (RRD -> MoCap)
    print("\n[4] Computing transformation (RRD -> MoCap)...")
    R, t, scale, rmse = compute_trajectory_transformation(
        traj_rrd_resampled,
        traj_mocap_resampled,
        allow_scaling=True
    )

    # Apply transformation
    print("\n[5] Applying transformation...")
    traj_rrd_aligned = apply_transformation(traj_rrd_resampled, R, t, scale)

    # Visualize alignment
    print("\n[6] Visualizing alignment...")
    visualize_trajectory_alignment(
        traj_rrd_resampled,
        traj_mocap_resampled,
        traj_rrd_aligned,
        R, t, scale, rmse,
        source_label="RRD (SLAM)",
        target_label="MoCap (Ground Truth)"
    )


if __name__ == "__main__":
    main()
