from fgr import fgr
from kissmatcher import registrate_point_clouds
from kissmatcher import bag2np

import os
import subprocess
import open3d as o3d
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

swift_front_lidar_topic = "/ec_swift/front_lidar_sensor/points_raw"
swift_back_lidar_topic = "/ec_swift/back_lidar_sensor/points_raw"
ouster_lidar_topic = "/sensor_station/lidar/points"
telemax_lidar_topic = ""

swift_mocap_topic = "/qualisys/TUDA_swift/pose"
ouster_mocap_topic = "/qualisys/TUDA_ouster/pose"
telemax_mocap_topic = ""

swift_mocap_offset = np.array([
         [0.5453810259462297, -0.8020423573026362, 0.24349043847812943, 4.638115830002652],
         [0.7342469812568727, 0.31703428250476934, -0.6003087824044075, 5.404119873519583],
         [0.4042782544894191, 0.5061791390497241, 0.7618016619421807, 0.2895704602735417],
         [0.0, 0.0, 0.0, 1.0],
     ])

mocap_ouster_offset = np.array([
    [ 0.51303887,  0.23690237,  1.33946109,  0.        ],
    [-0.23838657,  7.9347167 ,  0.13307461,  0.        ],
    [-5.13675547,  0.23512125, -0.11965466,  0.        ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])


swift_front_lidar_frame =np.array([
    [-np.sqrt(3)/2,  0.0, -0.5,  0.19],
    [ 0.0,          -1.0,  0.0,  0.0],
    [ 0.5,           0.0,  np.sqrt(3)/2, 0.2204],
    [ 0.0,           0.0,  0.0,  1.0]
])

swift_back_lidar_frame =np.array([
    [ np.sqrt(3)/2,  0.0, -0.5, -0.19],
    [ 0.0,           1.0,  0.0,  0.0],
    [ 0.5,           0.0,  np.sqrt(3)/2, 0.2204],
    [ 0.0,           0.0,  0.0,  1.0]
])


def swift_merge_front_back(front, back):
    merged = front.copy()
    return merged


def transform_point_cloud(points, transformation_matrix):
    """
    Transform a point cloud using a 4x4 transformation matrix.

    Args:
        points: np.array of shape (n, 3) representing n 3D points
        transformation_matrix: np.array of shape (4, 4) representing the transformation

    Returns:
        transformed_points: np.array of shape (n, 3) with transformed points
    """
    # Validate inputs
    assert points.shape[1] == 3, "Points must have shape (n, 3)"
    assert transformation_matrix.shape == (4, 4), "Transformation matrix must be 4x4"

    # Convert points to homogeneous coordinates (n, 4) by adding column of ones
    n_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones((n_points, 1))])

    # Apply transformation: (4, 4) @ (n, 4).T = (4, n)
    transformed_homogeneous = (transformation_matrix @ homogeneous_points.T).T

    # Convert back to 3D by dropping the homogeneous coordinate
    transformed_points = transformed_homogeneous[:, :3]

    return transformed_points


def get_robusticp_result(src_np, tgt_np):
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_np)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_np)

    # Save to PLY file
    o3d.io.write_point_cloud("src.ply", src)
    o3d.io.write_point_cloud("tgt.ply", tgt)
    subprocess.run(["./robusticpstuff/FRICP", "./target.ply", "./source.ply", "./robusticpstuff/results/", "3"])

    return np.loadtxt("/results/m3trans.txt")


def to_transformation_matrix(rotation_matrix, translation_vector):
    """
    Convert rotation matrix and translation vector to 4x4 homogeneous transformation matrix.

    Parameters:
    -----------
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix
    translation_vector : numpy.ndarray
        3x1 or (3,) translation vector

    Returns:
    --------
    numpy.ndarray
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector

    return T

def registrate_with_multiple_approaches(src, tgt):
    kissmatcher_result = registrate_point_clouds(src, tgt)
    fgr_result = fgr(src, tgt)
    print(fgr_result.transformation)
    #robusticp_result = get_robusticp_result(src, tgt)
    print(kissmatcher_result)
    gt_result = get_initial_transform_mcap("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap", "/qualisys/TUDA_swift/pose", "/qualisys/TUDA_ouster/pose")
    eval_kis = visualize_transformation_error(gt_result, to_transformation_matrix(kissmatcher_result[0], kissmatcher_result[1]), 1.0)
    print(eval_kis)
    eval_fgr = visualize_transformation_error(gt_result, fgr_result.transformation, 1.0)
    print(eval_fgr)

def eval(snapshot_dir):
    for file in os.listdir(snapshot_dir):
        if 'swift' in file and 'ouster' in file:
            # swift ouster
            src_front = bag2np(os.path.join(snapshot_dir, file), swift_front_lidar_topic, is_repetitive=False)
            src_back = bag2np(os.path.join(snapshot_dir, file), swift_back_lidar_topic, is_repetitive=False)
            src = swift_merge_front_back(src_front, src_back)
            tgt = bag2np(os.path.join(snapshot_dir, file), ouster_lidar_topic)
            registrate_with_multiple_approaches(src, tgt)
        """
        if 'swift' in file and 'telemax' in file:

        if 'ouster' in file and 'telemax' in file:

        if 'swift' in file and 'telemax' in file and 'ouster' in file:
"""

from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R

def get_initial_transform_mcap(bag_path, pose_src_topic, pose_tgt_topic):
    """
    Reads the first message from two topics in an MCAP bag and
    calculates the transform (Rotation/Translation) from src to tgt.

    Returns:
    --------
    numpy.ndarray
        4x4 homogeneous transformation matrix from src to tgt
    """
    poses = {pose_src_topic: None, pose_tgt_topic: None}

    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        # Iterate through messages
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            topic = channel.topic

            # Store the first message we see for each requested topic
            if topic == pose_src_topic or topic == pose_tgt_topic:
                poses[topic] = ros_msg.pose

            # Exit early once both poses are captured
            if poses[pose_src_topic] and poses[pose_tgt_topic]:
                break

    if not poses[pose_src_topic] or not poses[pose_tgt_topic]:
        raise ValueError("Could not find both topics in the provided bag.")

    # --- Calculation ---
    p_s = poses[pose_src_topic]
    p_t = poses[pose_tgt_topic]



    # Convert positions to numpy vectors
    pos_s = np.array([p_s.position.x, p_s.position.y, p_s.position.z])
    pos_t = np.array([p_t.position.x, p_t.position.y, p_t.position.z])

    # Convert orientations to SciPy Rotation objects
    # Note: ROS quaternions are [x, y, z, w], which SciPy accepts
    quat_s = [p_s.orientation.x, p_s.orientation.y, p_s.orientation.z, p_s.orientation.w]
    quat_t = [p_t.orientation.x, p_t.orientation.y, p_t.orientation.z, p_t.orientation.w]

    r_s = R.from_quat(quat_s)
    r_t = R.from_quat(quat_t)

    # 1. Rotation from src to tgt: R_rel = R_t * inv(R_s)
    rel_rot = r_t * r_s.inv()

    # 2. Translation from src to tgt in the world frame: t_rel = t_t - t_s
    rel_trans = pos_t - pos_s

    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rel_rot.as_matrix()
    T[:3, 3] = rel_trans

    return T
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


def visualize_transformation_error(T_gt, T_est, arrow_length=1.0):
    """
    Visualize rotation and translation errors between ground truth and estimated transformations.

    Parameters:
    -----------
    T_gt : numpy.ndarray
        4x4 ground truth transformation matrix
    T_est : numpy.ndarray
        4x4 estimated transformation matrix
    arrow_length : float
        Length of coordinate frame arrows for visualization
    """
    # Extract rotation matrices and translation vectors
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]

    # Calculate rotation error (relative rotation)
    R_error = R_gt.T @ R_est
    rot_error = Rotation.from_matrix(R_error)
    angle_error = np.degrees(rot_error.magnitude())
    axis_error = rot_error.as_rotvec()
    if np.linalg.norm(axis_error) > 1e-6:
        axis_error = axis_error / np.linalg.norm(axis_error)

    # Calculate translation error
    t_error = t_est - t_gt
    translation_error_magnitude = np.linalg.norm(t_error)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 5))

    # --- Subplot 1: 3D Visualization of Coordinate Frames ---
    ax1 = fig.add_subplot(131, projection='3d')

    def plot_frame(ax, R, t, label, colors, alpha=1.0):
        """Plot a coordinate frame."""
        origin = t
        for i, (color, axis_label) in enumerate(zip(colors, ['X', 'Y', 'Z'])):
            direction = R[:, i] * arrow_length
            ax.quiver(origin[0], origin[1], origin[2],
                      direction[0], direction[1], direction[2],
                      color=color, alpha=alpha, arrow_length_ratio=0.3,
                      linewidth=2, label=f'{label}-{axis_label}')

    # Plot ground truth frame (solid colors)
    plot_frame(ax1, R_gt, t_gt, 'GT', ['red', 'green', 'blue'], alpha=1.0)

    # Plot estimated frame (lighter colors)
    plot_frame(ax1, R_est, t_est, 'Est', ['salmon', 'lightgreen', 'lightblue'], alpha=0.7)

    # Plot translation error vector
    ax1.plot([t_gt[0], t_est[0]], [t_gt[1], t_est[1]], [t_gt[2], t_est[2]],
             'k--', linewidth=2, label='Translation Error')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Coordinate Frames Comparison')
    ax1.legend(loc='upper left', fontsize=8)

    # Set equal aspect ratio
    all_points = np.vstack([t_gt, t_est])
    max_range = np.max(np.ptp(all_points, axis=0)) * 0.5 + arrow_length
    mid = np.mean(all_points, axis=0)
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # --- Subplot 2: Rotation Error ---
    ax2 = fig.add_subplot(132)

    # Bar chart for rotation error
    ax2.bar(['Rotation Error'], [angle_error], color='coral', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Angle (degrees)', fontsize=12)
    ax2.set_title(f'Rotation Error: {angle_error:.3f}°', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add text with rotation axis
    if np.linalg.norm(axis_error) > 1e-6:
        ax2.text(0, angle_error * 0.5,
                 f'Axis: [{axis_error[0]:.2f}, {axis_error[1]:.2f}, {axis_error[2]:.2f}]',
                 ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Subplot 3: Translation Error ---
    ax3 = fig.add_subplot(133)

    # Component-wise translation error
    components = ['X', 'Y', 'Z', 'Total']
    errors = [t_error[0], t_error[1], t_error[2], translation_error_magnitude]
    colors_bar = ['red', 'green', 'blue', 'purple']

    bars = ax3.bar(components, np.abs(errors), color=colors_bar, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Error Magnitude', fontsize=12)
    ax3.set_title(f'Translation Error: {translation_error_magnitude:.3f}',
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{error:+.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print numerical summary
    print("=" * 60)
    print("TRANSFORMATION ERROR SUMMARY")
    print("=" * 60)
    print(f"\nRotation Error:")
    print(f"  Angle: {angle_error:.4f} degrees ({np.radians(angle_error):.4f} radians)")
    if np.linalg.norm(axis_error) > 1e-6:
        print(f"  Axis:  [{axis_error[0]:.4f}, {axis_error[1]:.4f}, {axis_error[2]:.4f}]")
    print(f"\nTranslation Error:")
    print(f"  X: {t_error[0]:+.4f}")
    print(f"  Y: {t_error[1]:+.4f}")
    print(f"  Z: {t_error[2]:+.4f}")
    print(f"  Magnitude: {translation_error_magnitude:.4f}")
    print("=" * 60)

    return {
        'rotation_error_deg': angle_error,
        'rotation_error_rad': np.radians(angle_error),
        'rotation_axis': axis_error,
        'translation_error': t_error,
        'translation_error_magnitude': translation_error_magnitude
    }


eval("/home/aaron/uni/thesis/data/snapshots")