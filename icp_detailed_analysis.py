"""
Detailed ICP Analysis with transformation output and comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from traj_extractor import (
    extract_trajectory_from_bags,
    get_rerun_entity_data,
)
from trajectory_transform import extract_trajectory_from_dataframe


def icp_detailed(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
):
    """
    ICP with detailed iteration tracking for analysis.
    Returns transformation and convergence history.
    """
    P = source[:, :3].astype(np.float64).copy()
    Q = target[:, :3].astype(np.float64)
    P_original = source[:, :3].astype(np.float64).copy()

    R_total = np.eye(3)
    t_total = np.zeros(3)

    tree = KDTree(Q)

    # History tracking
    history = {
        'rmse': [],
        'mean_error': [],
        'max_error': [],
        'rotation_angle': [],
        'translation_magnitude': []
    }

    for i in range(max_iterations):
        # Find closest points
        distances, indices = tree.query(P)
        Q_matched = Q[indices]

        # Track errors
        rmse = np.sqrt((distances ** 2).mean())
        history['rmse'].append(rmse)
        history['mean_error'].append(distances.mean())
        history['max_error'].append(distances.max())

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

        # Track transformation changes
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        history['rotation_angle'].append(np.degrees(angle))
        history['translation_magnitude'].append(np.linalg.norm(t))

        # Update total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        # Apply transformation
        P = (P @ R.T) + t

        # Check convergence
        if i > 0 and abs(history['rmse'][-2] - history['rmse'][-1]) < tolerance:
            break

    # Final aligned trajectory
    aligned = (P_original @ R_total.T) + t_total

    return R_total, t_total, aligned, history, i + 1


def rotation_to_euler(R):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    rot = Rotation.from_matrix(R)
    euler = rot.as_euler('xyz', degrees=True)
    return euler


def rotation_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation."""
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle > 1e-10:
        axis = rotvec / angle
    else:
        axis = np.array([0, 0, 1])
    return axis, np.degrees(angle)


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


def plot_detailed_icp_analysis(
    source: np.ndarray,
    target: np.ndarray,
    aligned: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    history: dict,
    iterations: int
):
    """Create comprehensive visualization of ICP results."""

    fig = plt.figure(figsize=(20, 16))

    # 1. 3D Before/After comparison
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    ax1.plot(source[:, 0], source[:, 1], source[:, 2], 'b-', linewidth=0.8, alpha=0.7, label='Source (RRD)')
    ax1.plot(target[:, 0], target[:, 1], target[:, 2], 'r-', linewidth=0.8, alpha=0.7, label='Target (MoCap)')
    ax1.set_title('Before ICP Alignment', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    _set_equal_axes(ax1, source, target)

    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    ax2.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2], 'b-', linewidth=0.8, alpha=0.7, label='Aligned (RRD)')
    ax2.plot(target[:, 0], target[:, 1], target[:, 2], 'r-', linewidth=0.8, alpha=0.7, label='Target (MoCap)')
    ax2.set_title(f'After ICP Alignment\nRMSE: {history["rmse"][-1]:.4f}m', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    _set_equal_axes(ax2, aligned, target)

    # 3. Overlay view
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    ax3.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2], 'b-', linewidth=1, alpha=0.8)
    ax3.plot(target[:, 0], target[:, 1], target[:, 2], 'r--', linewidth=1, alpha=0.6)
    ax3.set_title('Overlay (Aligned vs Target)', fontsize=12, fontweight='bold')
    _set_equal_axes(ax3, aligned, target)

    # 4. XY Projection (Top-down)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(aligned[:, 0], aligned[:, 1], 'b-', linewidth=1, label='Aligned', alpha=0.8)
    ax4.plot(target[:, 0], target[:, 1], 'r-', linewidth=1, label='Target', alpha=0.8)
    ax4.scatter(aligned[0, 0], aligned[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax4.scatter(aligned[-1, 0], aligned[-1, 1], c='purple', s=100, marker='s', zorder=5, label='End')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('XY Projection (Top-Down)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # 5. XZ Projection (Side view)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(aligned[:, 0], aligned[:, 2], 'b-', linewidth=1, label='Aligned', alpha=0.8)
    ax5.plot(target[:, 0], target[:, 2], 'r-', linewidth=1, label='Target', alpha=0.8)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Z (m)')
    ax5.set_title('XZ Projection (Side View)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. YZ Projection (Front view)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(aligned[:, 1], aligned[:, 2], 'b-', linewidth=1, label='Aligned', alpha=0.8)
    ax6.plot(target[:, 1], target[:, 2], 'r-', linewidth=1, label='Target', alpha=0.8)
    ax6.set_xlabel('Y (m)')
    ax6.set_ylabel('Z (m)')
    ax6.set_title('YZ Projection (Front View)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # 7. Convergence plot - RMSE
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(history['rmse'], 'b-', linewidth=2, marker='o', markersize=3)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('RMSE (m)')
    ax7.set_title('ICP Convergence (RMSE)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=history['rmse'][-1], color='r', linestyle='--', alpha=0.5)
    ax7.annotate(f'Final: {history["rmse"][-1]:.4f}m',
                 xy=(len(history['rmse'])-1, history['rmse'][-1]),
                 xytext=(len(history['rmse'])*0.5, history['rmse'][-1]*1.2),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    # 8. Point-wise error distribution
    ax8 = fig.add_subplot(3, 3, 8)
    tree = KDTree(target)
    distances, _ = tree.query(aligned)
    ax8.hist(distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax8.axvline(x=distances.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {distances.mean():.4f}m')
    ax8.axvline(x=np.median(distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.4f}m')
    ax8.set_xlabel('Distance Error (m)')
    ax8.set_ylabel('Count')
    ax8.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # 9. Error along trajectory
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(distances, 'g-', linewidth=0.5, alpha=0.7)
    ax9.fill_between(range(len(distances)), 0, distances, alpha=0.3, color='green')
    ax9.axhline(y=distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.4f}m')
    ax9.axhline(y=np.percentile(distances, 95), color='orange', linestyle='--', label=f'95th %ile: {np.percentile(distances, 95):.4f}m')
    ax9.set_xlabel('Point Index')
    ax9.set_ylabel('Error (m)')
    ax9.set_title('Error Along Trajectory', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/icp_detailed_analysis.svg', format='svg', bbox_inches='tight')
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/icp_detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Additional convergence analysis plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['rmse'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('RMSE (m)')
    axes[0, 0].set_title('RMSE Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['mean_error'], 'g-', linewidth=2, marker='s', markersize=4, label='Mean')
    axes[0, 1].plot(history['max_error'], 'r-', linewidth=2, marker='^', markersize=4, label='Max')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].set_title('Mean/Max Error Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['rotation_angle'], 'm-', linewidth=2, marker='d', markersize=4)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Rotation Change (degrees)')
    axes[1, 0].set_title('Per-Iteration Rotation')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['translation_magnitude'], 'c-', linewidth=2, marker='p', markersize=4)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Translation Magnitude (m)')
    axes[1, 1].set_title('Per-Iteration Translation')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/icp_convergence_analysis.svg', format='svg', bbox_inches='tight')
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/icp_convergence_analysis.png', dpi=150, bbox_inches='tight')
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


def print_transformation(R: np.ndarray, t: np.ndarray):
    """Print detailed transformation information."""
    print("\n" + "=" * 70)
    print("ICP TRANSFORMATION RESULT")
    print("=" * 70)

    print("\n[ROTATION MATRIX (3x3)]")
    print("-" * 40)
    for i in range(3):
        print(f"  [{R[i, 0]:12.8f}, {R[i, 1]:12.8f}, {R[i, 2]:12.8f}]")

    print("\n[TRANSLATION VECTOR]")
    print("-" * 40)
    print(f"  t = [{t[0]:12.8f}, {t[1]:12.8f}, {t[2]:12.8f}]")
    print(f"  |t| = {np.linalg.norm(t):.6f} m")

    # Euler angles
    euler = rotation_to_euler(R)
    print("\n[EULER ANGLES (XYZ convention)]")
    print("-" * 40)
    print(f"  Roll (X):  {euler[0]:10.4f} degrees")
    print(f"  Pitch (Y): {euler[1]:10.4f} degrees")
    print(f"  Yaw (Z):   {euler[2]:10.4f} degrees")

    # Axis-angle
    axis, angle = rotation_to_axis_angle(R)
    print("\n[AXIS-ANGLE REPRESENTATION]")
    print("-" * 40)
    print(f"  Rotation axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
    print(f"  Rotation angle: {angle:.4f} degrees")

    # Quaternion
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # [x, y, z, w]
    print("\n[QUATERNION (x, y, z, w)]")
    print("-" * 40)
    print(f"  q = [{quat[0]:.8f}, {quat[1]:.8f}, {quat[2]:.8f}, {quat[3]:.8f}]")

    # 4x4 Transformation matrix
    print("\n[HOMOGENEOUS TRANSFORMATION MATRIX (4x4)]")
    print("-" * 50)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    for i in range(4):
        print(f"  [{T[i, 0]:12.8f}, {T[i, 1]:12.8f}, {T[i, 2]:12.8f}, {T[i, 3]:12.8f}]")

    print("\n[COPY-PASTE FORMATS]")
    print("-" * 50)
    print("\nNumPy array (R):")
    print(f"R = np.array([")
    for i in range(3):
        print(f"    [{R[i, 0]}, {R[i, 1]}, {R[i, 2]}],")
    print("])")

    print("\nNumPy array (t):")
    print(f"t = np.array([{t[0]}, {t[1]}, {t[2]}])")

    print("\nHomogeneous matrix (T):")
    print(f"T = np.array([")
    for i in range(4):
        print(f"    [{T[i, 0]}, {T[i, 1]}, {T[i, 2]}, {T[i, 3]}],")
    print("])")

    print("\n" + "=" * 70)


def main():
    # Paths
    rrd_path = "/home/aaron/PycharmProjects/PythonProject/data.rrd"
    mocap_bag_path = "/home/aaron/uni/thesis/data/seq4/seq4_mocap"
    mocap_topic = "/qualisys/TUDA_swift/pose"

    print("=" * 60)
    print("Detailed ICP Analysis")
    print("=" * 60)

    # Extract trajectories
    print("\n[1] Extracting trajectories...")
    result = get_rerun_entity_data(rrd_path, "/world/trajectory", timeline="data_time")
    traj_rrd = extract_trajectory_from_dataframe(result["data"])
    print(f"    RRD trajectory: {traj_rrd.shape}")

    traj_mocap = extract_trajectory_from_bags(mocap_bag_path, mocap_topic)
    traj_mocap_xyz = traj_mocap[:, :3]
    print(f"    MoCap trajectory: {traj_mocap_xyz.shape}")

    # Resample
    print("\n[2] Resampling trajectories...")
    num_points = min(len(traj_rrd), len(traj_mocap_xyz))
    source = resample_trajectory(traj_rrd, num_points)
    target = resample_trajectory(traj_mocap_xyz, num_points)
    print(f"    Resampled to {num_points} points each")

    # Run ICP with detailed tracking
    print("\n[3] Running ICP algorithm...")
    R, t, aligned, history, iterations = icp_detailed(source, target, max_iterations=100)
    print(f"    Converged in {iterations} iterations")
    print(f"    Final RMSE: {history['rmse'][-1]:.6f} m")

    # Print transformation
    print_transformation(R, t)

    # Error statistics
    tree = KDTree(target)
    distances, _ = tree.query(aligned)
    print("\n[ERROR STATISTICS]")
    print("-" * 40)
    print(f"  RMSE:         {np.sqrt((distances**2).mean()):.6f} m")
    print(f"  Mean Error:   {distances.mean():.6f} m")
    print(f"  Median Error: {np.median(distances):.6f} m")
    print(f"  Max Error:    {distances.max():.6f} m")
    print(f"  Min Error:    {distances.min():.6f} m")
    print(f"  Std Dev:      {distances.std():.6f} m")
    print(f"  95th %ile:    {np.percentile(distances, 95):.6f} m")

    # Generate plots
    print("\n[4] Generating visualizations...")
    plot_detailed_icp_analysis(source, target, aligned, R, t, history, iterations)

    print("\nPlots saved to thesis_resources/:")
    print("  - icp_detailed_analysis.svg / .png")
    print("  - icp_convergence_analysis.svg / .png")


if __name__ == "__main__":
    main()
