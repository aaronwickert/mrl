"""
Point Cloud Registration Evaluation: Swift vs Ouster

Evaluates KISSMatcher and FGR (Fast Global Registration) algorithms
for aligning point clouds from Swift robot and Ouster sensor station.
Compares against MoCap ground truth.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import open3d as o3d

import subprocess
from fgr import fgr, bag2np as fgr_bag2np
from kissmatcher import registrate_point_clouds, bag2np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from overlap_calculator import compute_point_cloud_overlap

# FRICP paths
FRICP_EXECUTABLE = "/home/aaron/uni/thesis/robust_icp/Fast-Robust-ICP/build/FRICP"
FRICP_RESULTS_DIR = "/home/aaron/uni/thesis/robust_icp/Fast-Robust-ICP/build/data/res/"
FRICP_SRC_PLY = "/home/aaron/PycharmProjects/PythonProject/robusticpstuff/src.ply"
FRICP_TGT_PLY = "/home/aaron/PycharmProjects/PythonProject/robusticpstuff/tgt.ply"


# =============================================================================
# Configuration
# =============================================================================

# LiDAR Topics
SWIFT_FRONT_LIDAR_TOPIC = "/ec_swift/front_lidar_sensor/points_raw"
SWIFT_BACK_LIDAR_TOPIC = "/ec_swift/back_lidar_sensor/points_raw"
OUSTER_LIDAR_TOPIC = "/sensor_station/lidar/points"

# MoCap Topics
SWIFT_MOCAP_TOPIC = "/qualisys/TUDA_swift/pose"
OUSTER_MOCAP_TOPIC = "/qualisys/TUDA_ouster/pose"

# Transformation from SLAM to MoCap frame (computed via ICP)
SWIFT_MOCAP_OFFSET = np.array([
    [0.5453810259462297, -0.8020423573026362, 0.24349043847812943, 0],#4.638115830002652],
    [0.7342469812568727, 0.31703428250476934, -0.6003087824044075, 0],#5.404119873519583],
    [0.4042782544894191, 0.5061791390497241, 0.7618016619421807, 0,],#0.2895704602735417],
    [0.0, 0.0, 0.0, 1.0],
])

# Swift LiDAR frame transforms (body to lidar)
SWIFT_FRONT_LIDAR_FRAME = np.array([
    [-np.sqrt(3)/2,  0.0, -0.5,  0.19],
    [ 0.0,          -1.0,  0.0,  0.0],
    [ 0.5,           0.0,  np.sqrt(3)/2, 0.2204],
    [ 0.0,           0.0,  0.0,  1.0]
])

SWIFT_BACK_LIDAR_FRAME = np.array([
    [ np.sqrt(3)/2,  0.0, -0.5, -0.19],
    [ 0.0,           1.0,  0.0,  0.0],
    [ 0.5,           0.0,  np.sqrt(3)/2, 0.2204],
    [ 0.0,           0.0,  0.0,  1.0]
])

# Output directory
OUTPUT_DIR = "/home/aaron/PycharmProjects/PythonProject/thesis_resources"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegistrationResult:
    """Result from a registration algorithm."""
    name: str
    transformation: np.ndarray  # 4x4 homogeneous
    rotation_error_deg: float
    translation_error: float  # Translation vector error (direction + magnitude)
    translation_magnitude_error: float = 0.0  # Error in translation magnitude only
    computation_time: float = 0.0
    inliers: int = 0


@dataclass
class EvaluationResult:
    """Complete evaluation result for a snapshot."""
    snapshot_name: str
    ground_truth: np.ndarray
    results: List[RegistrationResult]
    src_points: int
    tgt_points: int
    overlap_ratio: float = 0.0
    feature_density_src: float = 0.0
    feature_density_tgt: float = 0.0


# =============================================================================
# Utility Functions
# =============================================================================

def to_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert rotation matrix and translation vector to 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def pose_to_transform_matrix(pose):
    """
    Convert a ROS pose to a 4x4 homogeneous transformation matrix.

    Parameters:
    -----------
    pose : geometry_msgs.msg.Pose
        ROS pose message containing position (x, y, z) and orientation (quaternion)

    Returns:
    --------
    numpy.ndarray
        4x4 homogeneous transformation matrix
    """
    # Extract position
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z

    # Extract quaternion (x, y, z, w)
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w

    # Normalize quaternion
    norm = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Convert quaternion to rotation matrix
    # Using the formula from quaternion to rotation matrix conversion
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R  # Rotation part
    T[:3, 3] = [x, y, z]  # Translation part

    return T

def transform_point_cloud(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform point cloud using 4x4 transformation matrix."""
    assert points.shape[1] == 3
    assert T.shape == (4, 4)
    n = points.shape[0]
    homogeneous = np.hstack([points, np.ones((n, 1))])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :3]


def compute_transformation_error(T_gt: np.ndarray, T_est: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute rotation and translation error between ground truth and estimated transforms.

    Returns:
        rotation_error_deg: Rotation error in degrees
        translation_error: Translation error magnitude (norm of difference vector)
        translation_magnitude_error: Error in translation magnitude (difference of norms)
        t_error_vector: Translation error vector [dx, dy, dz]
    """
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]

    # Rotation error
    R_error = R_gt.T @ R_est
    rot = Rotation.from_matrix(R_error)
    rotation_error_deg = np.degrees(rot.magnitude())

    # Translation error (vector difference)
    t_error = t_est - t_gt
    translation_error = np.linalg.norm(t_error)

    # Translation magnitude error (difference between magnitudes)
    t_gt_magnitude = np.linalg.norm(t_gt)
    t_est_magnitude = np.linalg.norm(t_est)
    translation_magnitude_error = t_est_magnitude - t_gt_magnitude

    return rotation_error_deg, translation_error, translation_magnitude_error, t_error

def get_position(bag_path: str, src_topic: str, tgt_topic: str) -> np.ndarray:
    """
    Get ground truth transformation from MoCap poses.
    Returns 4x4 transformation matrix from src to tgt.
    """
    poses = {src_topic: None, tgt_topic: None}

    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            topic = channel.topic

            if topic == src_topic:
                poses[src_topic] = ros_msg.pose
            elif topic == tgt_topic:
                poses[tgt_topic] = ros_msg.pose

            if poses[src_topic] and poses[tgt_topic]:
                break

    if not poses[src_topic] or not poses[tgt_topic]:
        raise ValueError(f"Could not find both topics in {bag_path}")

    p_s = poses[src_topic]
    p_t = poses[tgt_topic]
    return p_s

def get_ground_truth_transform(bag_path: str, src_topic: str, tgt_topic: str) -> np.ndarray:
    """
    Get ground truth transformation from MoCap poses.
    Returns 4x4 transformation matrix from src to tgt.
    """
    poses = {src_topic: None, tgt_topic: None}

    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            topic = channel.topic

            if topic == src_topic:
                poses[src_topic] = ros_msg.pose
            elif topic == tgt_topic:
                poses[tgt_topic] = ros_msg.pose

            if poses[src_topic] and poses[tgt_topic]:
                break

    if not poses[src_topic] or not poses[tgt_topic]:
        raise ValueError(f"Could not find both topics in {bag_path}")

    p_s = poses[src_topic]
    p_t = poses[tgt_topic]

    # Handle different message types
    if hasattr(p_s, 'pose'):
        p_s = p_s.pose
    if hasattr(p_t, 'pose'):
        p_t = p_t.pose

    pos_s = np.array([p_s.position.x, p_s.position.y, p_s.position.z])
    pos_t = np.array([p_t.position.x, p_t.position.y, p_t.position.z])

    quat_s = [p_s.orientation.x, p_s.orientation.y, p_s.orientation.z, p_s.orientation.w]
    quat_t = [p_t.orientation.x, p_t.orientation.y, p_t.orientation.z, p_t.orientation.w]

    r_s = Rotation.from_quat(quat_s)
    r_t = Rotation.from_quat(quat_t)

    # Relative rotation and translation
    rel_rot = r_t * r_s.inv()
    rel_trans = pos_t - pos_s

    return to_transformation_matrix(rel_rot.as_matrix(), rel_trans)


def merge_swift_lidars(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Merge front and back LiDAR point clouds from Swift robot."""
    # Transform each to body frame
    front_body = transform_point_cloud(front, np.linalg.inv(SWIFT_FRONT_LIDAR_FRAME))
    back_body = transform_point_cloud(back, np.linalg.inv(SWIFT_BACK_LIDAR_FRAME))
    return np.vstack([front_body, back_body])


def compute_feature_density(points: np.ndarray, voxel_size: float = 0.05) -> Tuple[float, int]:
    """
    Compute feature density metric for a point cloud.

    Uses FPFH (Fast Point Feature Histograms) feature extraction to identify
    geometrically distinctive keypoints.

    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape (N, 3)
    voxel_size : float
        Voxel size for downsampling and feature computation

    Returns
    -------
    phi : float
        Feature density ratio (N_features / N_points)
    n_features : int
        Number of detected features
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    # Count distinctive features (high variance in feature space)
    feature_array = np.asarray(fpfh.data).T  # Shape: (N_points, 33)

    # Features with high variance are considered distinctive
    feature_magnitudes = np.linalg.norm(feature_array, axis=1)
    threshold = np.percentile(feature_magnitudes, 75)  # Top 25% as features
    n_features = np.sum(feature_magnitudes > threshold)

    # Compute density relative to original point count
    n_total = len(points)
    phi = n_features / n_total if n_total > 0 else 0.0

    return phi, n_features


def compute_overlap_ratio(
    src: np.ndarray,
    tgt: np.ndarray,
    T_src_to_world: np.ndarray,
    T_tgt_to_world: np.ndarray,
    voxel_resolution: float = 0.1
) -> float:
    """
    Compute spatial overlap ratio between two point clouds.

    Parameters
    ----------
    src : np.ndarray
        Source point cloud (N, 3)
    tgt : np.ndarray
        Target point cloud (M, 3)
    T_src_to_world : np.ndarray
        4x4 transformation from source to world frame
    T_tgt_to_world : np.ndarray
        4x4 transformation from target to world frame
    voxel_resolution : float
        Voxel grid resolution in meters

    Returns
    -------
    rho : float
        Overlap ratio in [0, 1]
    """
    return compute_point_cloud_overlap(src, tgt, T_src_to_world, T_tgt_to_world, voxel_resolution)


# =============================================================================
# Registration Algorithms
# =============================================================================

def run_kissmatcher(src: np.ndarray, tgt: np.ndarray) -> RegistrationResult:
    """Run KISS-Matcher registration."""
    import time
    start = time.time()

    R, t = registrate_point_clouds(src, tgt)
    T = to_transformation_matrix(R, t)

    elapsed = time.time() - start

    return RegistrationResult(
        name="KISS-Matcher",
        transformation=T,
        rotation_error_deg=0,  # Will be filled later
        translation_error=0,
        computation_time=elapsed
    )


def run_fgr(src: np.ndarray, tgt: np.ndarray) -> RegistrationResult:
    """Run Fast Global Registration."""
    import time
    start = time.time()

    result = fgr(src, tgt)
    T = np.array(result.transformation)

    elapsed = time.time() - start

    return RegistrationResult(
        name="FGR",
        transformation=T,
        rotation_error_deg=0,
        translation_error=0,
        computation_time=elapsed
    )


def save_point_cloud_ply(points: np.ndarray, filename: str) -> None:
    """Save point cloud to PLY file for FRICP."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def run_fricp(src: np.ndarray, tgt: np.ndarray, method: int = 3) -> RegistrationResult:
    """
    Run FRICP (Fast Robust ICP) registration.

    Parameters
    ----------
    src : np.ndarray
        Source point cloud (N, 3)
    tgt : np.ndarray
        Target point cloud (M, 3)
    method : int
        FRICP method (0-4):
        0: ICP
        1: AA-ICP
        2: Sparse ICP (L1)
        3: Sparse ICP + GNC (Welsch)
        4: Sparse ICP + GNC (GM)

    Returns
    -------
    RegistrationResult
    """
    import time
    import os

    method_names = {
        0: "FRICP-ICP",
        1: "FRICP-AA",
        2: "FRICP-Sparse",
        3: "FRICP-Welsch",
        4: "FRICP-GM"
    }

    start = time.time()

    # Save point clouds to PLY
    save_point_cloud_ply(src, FRICP_SRC_PLY)
    save_point_cloud_ply(tgt, FRICP_TGT_PLY)

    T = np.eye(4)

    # Run FRICP
    try:
        result = subprocess.run(
            [FRICP_EXECUTABLE, FRICP_SRC_PLY, FRICP_TGT_PLY, FRICP_RESULTS_DIR, str(method)],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Try multiple ways to get the transformation
        trans_file = os.path.join(FRICP_RESULTS_DIR, f"m{method}trans.txt")
        reg_pc_file = os.path.join(FRICP_RESULTS_DIR, f"m{method}reg_pc.ply")

        if os.path.exists(trans_file):
            T = np.loadtxt(trans_file)
            if T.shape != (4, 4):
                T = np.eye(4)
        elif os.path.exists(reg_pc_file):
            # Extract transformation from registered point cloud using Kabsch
            print(f"    Extracting transformation from registered point cloud...")
            reg_pcd = o3d.io.read_point_cloud(reg_pc_file)
            reg_points = np.asarray(reg_pcd.points)

            # Use Kabsch algorithm to find transformation from src to registered
            if len(reg_points) == len(src):
                T = _kabsch_transformation(src, reg_points)
            else:
                print(f"    Warning: Point count mismatch ({len(src)} vs {len(reg_points)})")
        else:
            # Parse transformation from stdout if available
            stdout_lines = result.stdout.split('\n')
            for i, line in enumerate(stdout_lines):
                if 'transformation' in line.lower() or 'matrix' in line.lower():
                    try:
                        # Try to parse next 4 lines as transformation matrix
                        T = np.array([
                            [float(x) for x in stdout_lines[i+1].split()],
                            [float(x) for x in stdout_lines[i+2].split()],
                            [float(x) for x in stdout_lines[i+3].split()],
                            [float(x) for x in stdout_lines[i+4].split()]
                        ])
                        break
                    except:
                        continue

    except subprocess.TimeoutExpired:
        print("    FRICP timed out")
    except Exception as e:
        print(f"    FRICP error: {e}")

    elapsed = time.time() - start

    return RegistrationResult(
        name=method_names.get(method, f"FRICP-{method}"),
        transformation=T,
        rotation_error_deg=0,
        translation_error=0,
        computation_time=elapsed
    )


def _kabsch_transformation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute transformation matrix from P to Q using Kabsch algorithm."""
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

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_point_clouds(src: np.ndarray, tgt: np.ndarray,
                           T_gt: np.ndarray, results: List[RegistrationResult],
                           title: str = "Point Cloud Registration") -> None:
    """Visualize point clouds before and after registration with all algorithms."""

    n_methods = len(results) + 2  # +1 for original, +1 for ground truth
    n_cols = min(n_methods, 3)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    # Subsample for visualization
    max_pts = 5000
    src_vis = src[::max(1, len(src)//max_pts)]
    tgt_vis = tgt[::max(1, len(tgt)//max_pts)]

    plot_idx = 1

    # Original (unaligned)
    ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
    ax.scatter(src_vis[:, 0], src_vis[:, 1], src_vis[:, 2], c='blue', s=0.5, alpha=0.5, label='Source (Swift)')
    ax.scatter(tgt_vis[:, 0], tgt_vis[:, 1], tgt_vis[:, 2], c='red', s=0.5, alpha=0.5, label='Target (Ouster)')
    ax.set_title('Original (Unaligned)', fontweight='bold')
    ax.legend(fontsize=8, markerscale=5)
    _set_equal_axes_3d(ax, src_vis, tgt_vis)
    plot_idx += 1

    # Ground Truth alignment
    ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
    src_gt_aligned = transform_point_cloud(src_vis, T_gt)
    ax.scatter(src_gt_aligned[:, 0], src_gt_aligned[:, 1], src_gt_aligned[:, 2],
               c='green', s=0.5, alpha=0.5, label='GT Aligned')
    ax.scatter(tgt_vis[:, 0], tgt_vis[:, 1], tgt_vis[:, 2],
               c='red', s=0.5, alpha=0.5, label='Target')
    ax.set_title('Ground Truth (MoCap)\nReference Alignment', fontweight='bold', fontsize=10,
                 color='darkgreen')
    ax.legend(fontsize=8, markerscale=5)
    _set_equal_axes_3d(ax, src_gt_aligned, tgt_vis)
    plot_idx += 1

    # Each registration result
    for result in results:
        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')

        src_aligned = transform_point_cloud(src_vis, result.transformation)

        ax.scatter(src_aligned[:, 0], src_aligned[:, 1], src_aligned[:, 2],
                   c='blue', s=0.5, alpha=0.5, label='Aligned')
        ax.scatter(tgt_vis[:, 0], tgt_vis[:, 1], tgt_vis[:, 2],
                   c='red', s=0.5, alpha=0.5, label='Target')

        # Color code title based on error
        if result.rotation_error_deg < 10:
            title_color = 'darkgreen'
        elif result.rotation_error_deg < 45:
            title_color = 'orange'
        else:
            title_color = 'darkred'

        ax.set_title(f'{result.name}\nRot Err: {result.rotation_error_deg:.2f}°, '
                     f'Trans Err: {result.translation_error:.3f}m', fontweight='bold', fontsize=10,
                     color=title_color)
        ax.legend(fontsize=8, markerscale=5)
        _set_equal_axes_3d(ax, src_aligned, tgt_vis)
        plot_idx += 1

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pointcloud_registration.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/pointcloud_registration.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_transformation_comparison(T_gt: np.ndarray, results: List[RegistrationResult],
                                         arrow_length: float = 0.5) -> None:
    """Visualize and compare coordinate frames from different registration methods."""

    fig = plt.figure(figsize=(20, 10))

    # 3D coordinate frames comparison
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    def plot_frame(ax, T, label, colors, linestyle='-', alpha=1.0):
        R = T[:3, :3]
        t = T[:3, 3]
        for i, (color, axis_label) in enumerate(zip(colors, ['X', 'Y', 'Z'])):
            direction = R[:, i] * arrow_length
            ax.quiver(t[0], t[1], t[2], direction[0], direction[1], direction[2],
                      color=color, alpha=alpha, arrow_length_ratio=0.2, linewidth=2)

    # Ground truth frame
    plot_frame(ax1, T_gt, 'GT', ['darkred', 'darkgreen', 'darkblue'], alpha=1.0)

    # Algorithm frames
    algo_colors = [['salmon', 'lightgreen', 'lightblue'],
                   ['orange', 'lime', 'cyan']]
    for i, result in enumerate(results):
        plot_frame(ax1, result.transformation, result.name,
                   algo_colors[i % len(algo_colors)], alpha=0.7)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Coordinate Frames Comparison', fontweight='bold')

    # Create legend
    legend_elements = [Patch(facecolor='darkred', label='Ground Truth')]
    for i, result in enumerate(results):
        legend_elements.append(Patch(facecolor=algo_colors[i % len(algo_colors)][0],
                                     label=result.name))
    ax1.legend(handles=legend_elements, fontsize=9)

    # Rotation error comparison
    ax2 = fig.add_subplot(2, 3, 2)
    names = [r.name for r in results]
    rot_errors = [r.rotation_error_deg for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    bars = ax2.bar(names, rot_errors, color=colors, edgecolor='black')
    ax2.set_ylabel('Rotation Error (degrees)', fontsize=11)
    ax2.set_title('Rotation Error Comparison', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, err in zip(bars, rot_errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{err:.2f}°', ha='center', va='bottom', fontsize=10)

    # Translation error comparison
    ax3 = fig.add_subplot(2, 3, 3)
    trans_errors = [r.translation_error for r in results]

    bars = ax3.bar(names, trans_errors, color=colors, edgecolor='black')
    ax3.set_ylabel('Translation Error (m)', fontsize=11)
    ax3.set_title('Translation Error Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar, err in zip(bars, trans_errors):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{err:.3f}m', ha='center', va='bottom', fontsize=10)

    # Translation magnitude error comparison
    ax4 = fig.add_subplot(2, 3, 4)
    trans_mag_errors = [r.translation_magnitude_error for r in results]

    bar_colors = ['green' if e >= 0 else 'red' for e in trans_mag_errors]
    bars = ax4.bar(names, trans_mag_errors, color=bar_colors, edgecolor='black', alpha=0.7)
    ax4.set_ylabel('Translation Magnitude Error (m)', fontsize=11)
    ax4.set_title('Translation Magnitude Error\n(|t_est| - |t_gt|)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, err in zip(bars, trans_mag_errors):
        y_pos = bar.get_height() + 0.01 if err >= 0 else bar.get_height() - 0.02
        va = 'bottom' if err >= 0 else 'top'
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{err:+.4f}m', ha='center', va=va, fontsize=9)

    # Computation time comparison
    ax5 = fig.add_subplot(2, 3, 5)
    times = [r.computation_time for r in results]

    bars = ax5.bar(names, times, color=colors, edgecolor='black')
    ax5.set_ylabel('Computation Time (s)', fontsize=11)
    ax5.set_title('Computation Time Comparison', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    for bar, t in zip(bars, times):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{t:.2f}s', ha='center', va='bottom', fontsize=10)

    # Summary info panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    t_gt_mag = np.linalg.norm(T_gt[:3, 3])
    summary_text = f"""
    Summary
    {'='*35}

    Ground Truth:
    |t_gt| = {t_gt_mag:.4f} m

    Algorithms Compared: {len(results)}

    Translation Magnitude Error:
    • Positive (+) = overestimate
    • Negative (-) = underestimate

    Best results prefer values
    closest to zero.
    {'='*35}
    """
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/transformation_comparison.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/transformation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_error_analysis(T_gt: np.ndarray, results: List[RegistrationResult]) -> None:
    """Detailed error analysis visualization."""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    names = [r.name for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # Translation error components (X, Y, Z)
    ax = axes[0, 0]
    t_gt = T_gt[:3, 3]

    x = np.arange(len(results))
    width = 0.25

    dx = [r.transformation[0, 3] - t_gt[0] for r in results]
    dy = [r.transformation[1, 3] - t_gt[1] for r in results]
    dz = [r.transformation[2, 3] - t_gt[2] for r in results]

    ax.bar(x - width, dx, width, label='ΔX', color='red', alpha=0.7)
    ax.bar(x, dy, width, label='ΔY', color='green', alpha=0.7)
    ax.bar(x + width, dz, width, label='ΔZ', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Error (m)')
    ax.set_title('Translation Error Components', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Euler angle errors
    ax = axes[0, 1]
    R_gt = T_gt[:3, :3]
    euler_gt = Rotation.from_matrix(R_gt).as_euler('xyz', degrees=True)

    roll_err = []
    pitch_err = []
    yaw_err = []

    for r in results:
        euler_est = Rotation.from_matrix(r.transformation[:3, :3]).as_euler('xyz', degrees=True)
        roll_err.append(euler_est[0] - euler_gt[0])
        pitch_err.append(euler_est[1] - euler_gt[1])
        yaw_err.append(euler_est[2] - euler_gt[2])

    ax.bar(x - width, roll_err, width, label='ΔRoll', color='red', alpha=0.7)
    ax.bar(x, pitch_err, width, label='ΔPitch', color='green', alpha=0.7)
    ax.bar(x + width, yaw_err, width, label='ΔYaw', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Rotation Error Components (Euler)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Translation magnitude error
    ax = axes[0, 2]
    trans_mag_errors = [r.translation_magnitude_error for r in results]

    bar_colors = ['green' if e >= 0 else 'red' for e in trans_mag_errors]
    bars = ax.bar(names, trans_mag_errors, color=bar_colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Magnitude Error (m)')
    ax.set_title('Translation Magnitude Error\n(|t_est| - |t_gt|)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, err in zip(bars, trans_mag_errors):
        y_pos = bar.get_height() + 0.01 if err >= 0 else bar.get_height() - 0.02
        va = 'bottom' if err >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{err:+.4f}m', ha='center', va=va, fontsize=9)

    # Combined error metric
    ax = axes[0, 3]
    combined = [np.sqrt(r.rotation_error_deg**2 + (r.translation_error * 10)**2) for r in results]

    bars = ax.bar(names, combined, color=colors, edgecolor='black')
    ax.set_ylabel('Combined Error')
    ax.set_title('Combined Error Metric\n(√(rot² + (10×trans)²))', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Ground truth transformation visualization
    ax = axes[1, 0]
    ax.text(0.1, 0.9, 'Ground Truth Transformation:', fontsize=12, fontweight='bold',
            transform=ax.transAxes, verticalalignment='top')

    T_str = '\n'.join([f'[{T_gt[i,0]:8.4f}, {T_gt[i,1]:8.4f}, {T_gt[i,2]:8.4f}, {T_gt[i,3]:8.4f}]'
                       for i in range(4)])
    ax.text(0.1, 0.75, T_str, fontsize=10, family='monospace',
            transform=ax.transAxes, verticalalignment='top')

    euler = Rotation.from_matrix(R_gt).as_euler('xyz', degrees=True)
    ax.text(0.1, 0.35, f'Euler (XYZ): [{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]',
            fontsize=10, transform=ax.transAxes)
    ax.text(0.1, 0.25, f'Translation: [{t_gt[0]:.4f}, {t_gt[1]:.4f}, {t_gt[2]:.4f}]',
            fontsize=10, transform=ax.transAxes)
    ax.axis('off')

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [['Algorithm', 'Rot Err (°)', 'Trans Err (m)', 'Trans Mag Err (m)', 'Time (s)']]
    for r in results:
        table_data.append([r.name, f'{r.rotation_error_deg:.3f}',
                          f'{r.translation_error:.4f}', f'{r.translation_magnitude_error:+.4f}',
                          f'{r.computation_time:.3f}'])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.22, 0.18, 0.2, 0.22, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Header styling
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Results Summary', fontweight='bold', pad=20)

    # Winner annotation
    ax = axes[1, 2]
    best_rot = min(results, key=lambda r: r.rotation_error_deg)
    best_trans = min(results, key=lambda r: r.translation_error)
    best_trans_mag = min(results, key=lambda r: abs(r.translation_magnitude_error))
    best_time = min(results, key=lambda r: r.computation_time)

    ax.text(0.5, 0.85, 'Best Results', fontsize=14, fontweight='bold',
            transform=ax.transAxes, ha='center')
    ax.text(0.5, 0.68, f'Lowest Rotation Error: {best_rot.name} ({best_rot.rotation_error_deg:.3f}°)',
            fontsize=10, transform=ax.transAxes, ha='center')
    ax.text(0.5, 0.53, f'Lowest Translation Error: {best_trans.name} ({best_trans.translation_error:.4f}m)',
            fontsize=10, transform=ax.transAxes, ha='center')
    ax.text(0.5, 0.38, f'Lowest Trans Mag Error: {best_trans_mag.name} ({best_trans_mag.translation_magnitude_error:+.4f}m)',
            fontsize=10, transform=ax.transAxes, ha='center')
    ax.text(0.5, 0.23, f'Fastest: {best_time.name} ({best_time.computation_time:.3f}s)',
            fontsize=10, transform=ax.transAxes, ha='center')
    ax.axis('off')

    # Additional info panel
    ax = axes[1, 3]
    ax.axis('off')

    # Display ground truth translation magnitude for reference
    t_gt_mag = np.linalg.norm(T_gt[:3, 3])
    info_text = f"""
    Reference Values
    {'='*30}

    Ground Truth Translation:
    |t_gt| = {t_gt_mag:.4f} m

    Translation Magnitude Error:
    Positive = overestimate
    Negative = underestimate

    {'='*30}
    """
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/error_analysis.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def _set_equal_axes_3d(ax, *point_clouds):
    """Set equal axis scaling for 3D plot."""
    all_points = np.vstack([pc for pc in point_clouds])
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


def visualize_environment_metrics(eval_result: 'EvaluationResult') -> None:
    """Visualize environment metrics: overlap ratio and feature density."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overlap ratio gauge
    ax = axes[0]
    overlap = eval_result.overlap_ratio

    # Create a simple bar showing overlap
    ax.barh([0], [overlap], height=0.5, color='steelblue', alpha=0.8)
    ax.barh([0], [1-overlap], left=[overlap], height=0.5, color='lightgray', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Overlap Ratio (ρ)')
    ax.set_title(f'Point Cloud Overlap\nρ = {overlap:.4f} ({overlap*100:.1f}%)',
                 fontweight='bold', fontsize=12)

    # Add percentage markers
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Interpretation text
    if overlap < 0.1:
        interp = "Very Low - Registration likely to fail"
        color = 'red'
    elif overlap < 0.3:
        interp = "Low - Registration challenging"
        color = 'orange'
    elif overlap < 0.5:
        interp = "Moderate - Registration feasible"
        color = 'gold'
    else:
        interp = "Good - Registration favorable"
        color = 'green'
    ax.text(0.5, -0.35, interp, ha='center', fontsize=10, color=color,
            transform=ax.transAxes)

    # Feature density comparison
    ax = axes[1]
    densities = [eval_result.feature_density_src, eval_result.feature_density_tgt]
    labels = ['Swift\n(Source)', 'Ouster\n(Target)']
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(labels, densities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Feature Density (φ)')
    ax.set_title('Feature Density Comparison', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    for bar, d in zip(bars, densities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'φ = {d:.6f}', ha='center', va='bottom', fontsize=10)

    # Summary metrics panel
    ax = axes[2]
    ax.axis('off')

    summary_text = f"""
    Environment Metrics Summary
    {'='*40}

    Point Cloud Overlap (ρ)
    -----------------------
    ρ = {eval_result.overlap_ratio:.4f}
    Interpretation: {overlap*100:.1f}% spatial overlap

    Feature Density (φ)
    -------------------
    Source (Swift):  φ = {eval_result.feature_density_src:.6f}
    Target (Ouster): φ = {eval_result.feature_density_tgt:.6f}

    Point Counts
    ------------
    Source: {eval_result.src_points:,} points
    Target: {eval_result.tgt_points:,} points

    {'='*40}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Environment Metrics: {eval_result.snapshot_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/environment_metrics.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/environment_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_swift_ouster(snapshot_path: str) -> EvaluationResult:
    """
    Evaluate registration algorithms on Swift-Ouster point cloud pair.

    Args:
        snapshot_path: Path to .mcap file containing synchronized data

    Returns:
        EvaluationResult with all algorithm results
    """
    print("=" * 60)
    print(f"Evaluating: {os.path.basename(snapshot_path)}")
    print("=" * 60)

    # Extract point clouds
    print("\n[1] Extracting point clouds...")
    src_front = bag2np(snapshot_path, SWIFT_FRONT_LIDAR_TOPIC, is_repetitive=False)
    src_back = bag2np(snapshot_path, SWIFT_BACK_LIDAR_TOPIC, is_repetitive=False)
    src = src_front#merge_swift_lidars(src_front, src_back)
    tgt = bag2np(snapshot_path, OUSTER_LIDAR_TOPIC, is_repetitive=True)

    print(f"    Swift (merged): {src.shape[0]} points")
    print(f"    Ouster: {tgt.shape[0]} points")

    # Compute feature density metrics
    print("\n[2] Computing feature density metrics...")
    phi_src, n_feat_src = compute_feature_density(src)
    phi_tgt, n_feat_tgt = compute_feature_density(tgt)
    print(f"    Swift feature density:  φ = {phi_src:.6f} ({n_feat_src} features)")
    print(f"    Ouster feature density: φ = {phi_tgt:.6f} ({n_feat_tgt} features)")

    # Get ground truth from MoCap
    print("\n[3] Getting ground truth from MoCap...")
    T_gt = get_ground_truth_transform(snapshot_path, SWIFT_MOCAP_TOPIC, OUSTER_MOCAP_TOPIC)
    T_swift = pose_to_transform_matrix(get_position(snapshot_path, SWIFT_MOCAP_TOPIC, OUSTER_MOCAP_TOPIC)) + SWIFT_MOCAP_OFFSET
    T_ouster = pose_to_transform_matrix(get_position(snapshot_path, OUSTER_MOCAP_TOPIC, OUSTER_MOCAP_TOPIC))

    src = transform_point_cloud(src, T_swift)
    tgt = transform_point_cloud(tgt, T_ouster)

    print(f"    Ground truth transformation obtained")



    # Compute overlap ratio using ground truth poses
    print("\n[4] Computing point cloud overlap...")
    # Use identity for target (reference frame), and T_gt for source
    overlap = compute_overlap_ratio(src, tgt, T_gt, np.eye(4), voxel_resolution=0.1)
    print(f"    Overlap ratio: ρ = {overlap:.4f} ({overlap*100:.1f}%)")

    # Run registration algorithms
    print("\n[5] Running registration algorithms...")
    results = []

    # KISS-Matcher
    print("    - Running KISS-Matcher...")
    try:
        kiss_result = run_kissmatcher(src, tgt)
        rot_err, trans_err, trans_mag_err, _ = compute_transformation_error(T_gt, kiss_result.transformation)
        kiss_result.rotation_error_deg = rot_err
        kiss_result.translation_error = trans_err
        kiss_result.translation_magnitude_error = trans_mag_err
        results.append(kiss_result)
        print(f"      Rotation Error: {rot_err:.3f}°, Translation Error: {trans_err:.4f}m, Trans Mag Error: {trans_mag_err:.4f}m")
    except Exception as e:
        print(f"      KISS-Matcher failed: {e}")

    # FGR
    print("    - Running FGR...")
    try:
        fgr_result = run_fgr(src, tgt)
        rot_err, trans_err, trans_mag_err, _ = compute_transformation_error(T_gt, fgr_result.transformation)
        fgr_result.rotation_error_deg = rot_err
        fgr_result.translation_error = trans_err
        fgr_result.translation_magnitude_error = trans_mag_err
        results.append(fgr_result)
        print(f"      Rotation Error: {rot_err:.3f}°, Translation Error: {trans_err:.4f}m, Trans Mag Error: {trans_mag_err:.4f}m")
    except Exception as e:
        print(f"      FGR failed: {e}")

    # FRICP (Welsch robust kernel)
    print("    - Running FRICP (Welsch)...")
    try:
        fricp_result = run_fricp(src, tgt, method=3)
        rot_err, trans_err, trans_mag_err, _ = compute_transformation_error(T_gt, fricp_result.transformation)
        fricp_result.rotation_error_deg = rot_err
        fricp_result.translation_error = trans_err
        fricp_result.translation_magnitude_error = trans_mag_err
        results.append(fricp_result)
        print(f"      Rotation Error: {rot_err:.3f}°, Translation Error: {trans_err:.4f}m, Trans Mag Error: {trans_mag_err:.4f}m")
    except Exception as e:
        print(f"      FRICP failed: {e}")

    return EvaluationResult(
        snapshot_name=os.path.basename(snapshot_path),
        ground_truth=T_gt,
        results=results,
        src_points=src.shape[0],
        tgt_points=tgt.shape[0],
        overlap_ratio=overlap,
        feature_density_src=phi_src,
        feature_density_tgt=phi_tgt
    )


def evaluate_snapshot_directory(snapshot_dir: str) -> List[EvaluationResult]:
    """Evaluate all Swift-Ouster snapshots in a directory."""
    all_results = []

    for file in sorted(os.listdir(snapshot_dir)):
        snapshot_path = os.path.join(snapshot_dir, file)
        try:
            result = evaluate_swift_ouster(snapshot_path)
            all_results.append(result)

            # Visualize this snapshot
            src_front = bag2np(snapshot_path, SWIFT_FRONT_LIDAR_TOPIC, is_repetitive=False)
            src_back = bag2np(snapshot_path, SWIFT_BACK_LIDAR_TOPIC, is_repetitive=False)
            src = merge_swift_lidars(src_front, src_back)
            tgt = bag2np(snapshot_path, OUSTER_LIDAR_TOPIC, is_repetitive=True)

            print("\n[6] Generating visualizations...")
            visualize_point_clouds(src, tgt, result.ground_truth, result.results,
                                   title=f"Registration: {file}")
            visualize_transformation_comparison(result.ground_truth, result.results)
            visualize_error_analysis(result.ground_truth, result.results)
            visualize_environment_metrics(result)

        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def print_summary(results: List[EvaluationResult]) -> None:
    """Print summary of all evaluation results."""
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)

    for eval_result in results:
        print(f"\nSnapshot: {eval_result.snapshot_name}")
        print(f"  Points: Source={eval_result.src_points}, Target={eval_result.tgt_points}")
        print(f"\n  --- Environment Metrics ---")
        print(f"  Overlap Ratio (ρ):        {eval_result.overlap_ratio:.4f} ({eval_result.overlap_ratio*100:.1f}%)")
        print(f"  Feature Density (φ) Src:  {eval_result.feature_density_src:.6f}")
        print(f"  Feature Density (φ) Tgt:  {eval_result.feature_density_tgt:.6f}")
        print(f"\n  --- Registration Results ---")
        print(f"  {'Algorithm':<20} {'Rot Err (°)':<15} {'Trans Err (m)':<15} {'Trans Mag Err (m)':<18} {'Time (s)':<10}")
        print(f"  {'-'*78}")
        for r in eval_result.results:
            print(f"  {r.name:<20} {r.rotation_error_deg:<15.3f} {r.translation_error:<15.4f} {r.translation_magnitude_error:<+18.4f} {r.computation_time:<10.3f}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Evaluate snapshots
    snapshot_dir = "/home/aaron/uni/thesis/data/snapshots"

    if os.path.exists(snapshot_dir):
        results = evaluate_snapshot_directory(snapshot_dir)
        print("im here")
        print_summary(results)
    else:
        print(f"Snapshot directory not found: {snapshot_dir}")
        print("Please update the path or run with a specific snapshot file.")
