import numpy as np
from pathlib import Path
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import struct


def extract_trajectory_from_bags(bag_path: str, pose_topic: str):
    """
    Extract trajectory from ROS2 bags using MCAP.

    Args:
        bag_path: Path to the directory containing ROS2 bag files
        pose_topic: Name of the pose topic (e.g., '/robot_pose')

    Returns:
        numpy array of shape (N, 7) containing [x, y, z, qx, qy, qz, qw] for each pose
        or shape (N, 4) containing [x, y, z, timestamp] if you prefer
    """
    bag_dir = Path(bag_path)

    # Find all .mcap files in the bag directory
    mcap_files = sorted(bag_dir.glob("*.mcap"))

    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {bag_path}")

    poses = []
    decoder_factory = DecoderFactory()

    for mcap_file in mcap_files:
        print(f"Reading {mcap_file.name}...")

        with open(mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[decoder_factory])

            message_count = 0
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(
                    topics=[pose_topic]
            ):
                try:
                    # Handle different message types
                    if hasattr(ros_msg, 'pose'):
                        if hasattr(ros_msg.pose, 'pose'):
                            # PoseWithCovarianceStamped
                            pose = ros_msg.pose.pose
                        else:
                            # PoseStamped
                            pose = ros_msg.pose
                    elif hasattr(ros_msg, 'position'):
                        # Direct Pose message
                        pose = ros_msg
                    else:
                        print(f"Unknown message type: {type(ros_msg)}")
                        continue

                    # Extract position
                    x = pose.position.x
                    y = pose.position.y
                    z = pose.position.z

                    # Extract orientation (quaternion)
                    qx = pose.orientation.x
                    qy = pose.orientation.y
                    qz = pose.orientation.z
                    qw = pose.orientation.w

                    # Use message timestamp (nanoseconds)
                    timestamp = message.log_time / 1e9  # Convert to seconds

                    poses.append([x, y, z, qx, qy, qz, qw, timestamp])
                    message_count += 1

                except Exception as e:
                    print(f"Error processing message {message_count}: {e}")
                    continue

            print(f"Extracted {message_count} poses from {mcap_file.name}")

    if not poses:
        raise ValueError(f"No poses found for topic {pose_topic}")

    # Convert to numpy array
    trajectory = np.array(poses)

    print(f"\nTrajectory statistics:")
    print(f"Total poses: {len(trajectory)}")
    print(f"X range: [{trajectory[:, 0].min():.3f}, {trajectory[:, 0].max():.3f}]")
    print(f"Y range: [{trajectory[:, 1].min():.3f}, {trajectory[:, 1].max():.3f}]")
    print(f"Z range: [{trajectory[:, 2].min():.3f}, {trajectory[:, 2].max():.3f}]")
    print(f"Duration: {trajectory[-1, 7] - trajectory[0, 7]:.2f} seconds")

    return trajectory


# Alternative version that returns only position + timestamp
def extract_position_trajectory(bag_path: str, pose_topic: str):
    """
    Extract position trajectory from ROS2 bags using MCAP.

    Args:
        bag_path: Path to the directory containing ROS2 bag files
        pose_topic: Name of the pose topic (e.g., '/robot_pose')

    Returns:
        numpy array of shape (N, 4) containing [x, y, z, timestamp]
    """
    bag_dir = Path(bag_path)

    # Find all .mcap files in the bag directory
    mcap_files = sorted(bag_dir.glob("*.mcap"))

    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {bag_path}")

    poses = []
    decoder_factory = DecoderFactory()

    for mcap_file in mcap_files:
        print(f"Reading {mcap_file.name}...")

        with open(mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[decoder_factory])

            message_count = 0
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(
                    topics=[pose_topic]
            ):
                try:
                    # Handle different message types
                    if hasattr(ros_msg, 'pose'):
                        if hasattr(ros_msg.pose, 'pose'):
                            # PoseWithCovarianceStamped
                            pose = ros_msg.pose.pose
                        else:
                            # PoseStamped
                            pose = ros_msg.pose
                    elif hasattr(ros_msg, 'position'):
                        # Direct Pose message
                        pose = ros_msg
                    else:
                        print(f"Unknown message type: {type(ros_msg)}")
                        continue

                    # Extract position
                    x = pose.position.x
                    y = pose.position.y
                    z = pose.position.z

                    # Use message timestamp (nanoseconds)
                    timestamp = message.log_time / 1e9  # Convert to seconds

                    poses.append([x, y, z, timestamp])
                    message_count += 1

                except Exception as e:
                    print(f"Error processing message {message_count}: {e}")
                    continue

            print(f"Extracted {message_count} poses from {mcap_file.name}")

    if not poses:
        raise ValueError(f"No poses found for topic {pose_topic}")

    # Convert to numpy array
    trajectory = np.array(poses)

    print(f"\nTrajectory statistics:")
    print(f"Total poses: {len(trajectory)}")
    print(f"X range: [{trajectory[:, 0].min():.3f}, {trajectory[:, 0].max():.3f}]")
    print(f"Y range: [{trajectory[:, 1].min():.3f}, {trajectory[:, 1].max():.3f}]")
    print(f"Z range: [{trajectory[:, 2].min():.3f}, {trajectory[:, 2].max():.3f}]")
    print(f"Duration: {trajectory[-1, 3] - trajectory[0, 3]:.2f} seconds")

    return trajectory

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_equal_axes(traj: np.ndarray, pos_idx=(0, 1, 2)):
    """
    Plot a 3D trajectory with equal scaling on all axes.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array of shape (N, M)
    pos_idx : tuple
        Indices of (x, y, z) in traj
    """
    x = traj[:, pos_idx[0]]
    y = traj[:, pos_idx[1]]
    z = traj[:, pos_idx[2]]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory")

    # ---- enforce equal axis scaling ----
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # -----------------------------------

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory(df, column_name='/world/trajectory:LineStrips3D:strips'):
    """
    Plot 3D trajectory from a pandas DataFrame with equal axis scaling.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing trajectory data
    column_name : str
        Name of the column containing trajectory points
        Default: '/world/trajectory:LineStrips3D:strips'
    """
    # Extract all trajectory points
    all_points = []

    for idx, row in df.iterrows():
        strips = row[column_name]
        if len(strips) != 0:
            # Each strip is a list of [x, y, z] coordinates
            for strip in strips:
                for point in strip:
                    if len(point) == 3:
                        all_points.append(point)

    if not all_points:
        print("No valid trajectory points found")
        return

    # Convert to numpy array for easier manipulation
    points = np.array(all_points)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            'b-', linewidth=2, label='Trajectory')

    # Mark start and end points
    ax.scatter(points[0, 0], points[0, 1], points[0, 2],
               c='green', s=100, marker='o', label='Start')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2],
               c='red', s=100, marker='o', label='End')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Plot')
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


# Example usage:
# plot_trajectory(df)


import rerun as rr
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd


def get_rerun_entity_data(
        rrd_file_path: str,
        entity_path: str,
        timeline: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve data from a Rerun entity stored in an .rrd file.

    Args:
        rrd_file_path: Path to the .rrd file
        entity_path: The path to the entity (e.g., "/world/trajectory")
        timeline: Optional timeline name to use as index (e.g., "data_time", "log_time")
                 If not provided, will use the first available timeline

    Returns:
        Dictionary containing the entity's data and metadata

    Example:
        # Get data indexed by a timeline
        data = get_rerun_entity_data("data.rrd", "/world/trajectory")

        # Get data with a specific timeline
        data = get_rerun_entity_data("data.rrd", "/world/trajectory",
                                     timeline="data_time")
    """
    rrd_path = Path(rrd_file_path)

    if not rrd_path.exists():
        raise FileNotFoundError(f"RRD file not found: {rrd_file_path}")

    # Load the recording using the older dataframe API
    recording = rr.dataframe.load_recording(str(rrd_path))

    # Get schema to inspect available timelines
    schema = recording.schema()
    available_timeline_objs = schema.index_columns()

    # Extract timeline names
    available_timelines = [str(t).replace("Index(timeline:", "").replace(")", "")
                           for t in available_timeline_objs]

    # If no timeline specified, use the first available one
    if timeline is None:
        if len(available_timelines) > 0:
            timeline = available_timelines[0]

    # Create a view of the recording
    view = recording.view(index=timeline, contents=entity_path)

    # Select and read all data
    batch_reader = view.select()
    table = batch_reader.read_all()

    # Convert to pandas for easier manipulation
    df = table.to_pandas()

    result = {
        "entity_path": entity_path,
        "rrd_file": str(rrd_path),
        "timeline": timeline,
        "available_timelines": available_timelines,
        "data": df,
        "arrow_table": table,
        "num_rows": len(df),
        "column_names": list(df.columns),
    }

    return result


# Example usage
if __name__ == "__main__":
    # Extract the trajectory data
    result = get_rerun_entity_data(
        "/home/aaron/PycharmProjects/PythonProject/data.rrd",
        "/world/trajectory",
        timeline="data_time"
    )

    print(f"Timeline: {result['timeline']}")
    print(f"Available timelines: {result['available_timelines']}")
    print(f"Extracted {result['num_rows']} rows")
    print(f"Columns: {result['column_names']}")

    print("\nDataFrame preview:")
    print(result['data'].head())

    print(result['data'].to_numpy())

    print("\nDataFrame info:")
    print(result['data'].info())


# Example usage
if __name__ == "__main__":
    # Extract the trajectory data
    result = get_rerun_entity_data("/home/aaron/PycharmProjects/PythonProject/data.rrd",
                                   "/world/trajectory",
                                   timeline="data_time")

    print(f"Timeline: {result['timeline']}")
    print(f"Available timelines: {result['available_timelines']}")
    print(f"Extracted {result['num_rows']} rows")
    print(f"Columns: {result['column_names']}")

    # Convert to pandas if needed
    import pandas as pd
    plot_trajectory(result["data"], "/world/trajectory:LineStrips3D:strips")
    from trajectory_transform import compute_trajectory_transformation
    from trajectory_transform import extract_trajectory_from_dataframe
    np = extract_trajectory_from_dataframe(result["data"])

    trajectory = extract_trajectory_from_bags(
         "/home/aaron/uni/thesis/data/seq4/seq4_mocap",
         "/qualisys/TUDA_swift/pose"
     )

    result = compute_trajectory_transformation(trajectory, np)
    print(result)

    print(type(result))
    print("\nDataFrame preview:")
    #print(df.head())

"""

# Example usage
if __name__ == "__main__":
    # First, let's inspect what's in the file
    with rr.server.Server(datasets={"dataset": ["/home/aaron/PycharmProjects/PythonProject/data.rrd"]}) as server:
        client = server.client()
        dataset = client.get_dataset("dataset")
        schema = dataset.schema()

        print("Available timelines:", schema.index_columns())
        print("Available components:", schema.component_columns())

    # Then extract the data
    result = get_rerun_entity_data("data.rrd", "/world/trajectory")
    print(f"Extracted {result['num_rows']} rows")
    print(f"Columns: {result['column_names']}")


if __name__ == "__main__":
    rko_lio_solution = get_rerun_entity_data("/home/aaron/PycharmProjects/PythonProject/data.rrd" ,"/world/trajectory")
    print(rko_lio_solution.shape)
    pass
    trajectory = extract_trajectory_from_bags(
         "/home/aaron/uni/thesis/data/seq4/seq4_mocap",
         "/qualisys/TUDA_swift/pose"
     )
    print(f"Extracted {len(trajectory)} poses")
    print(f"Trajectory shape: {trajectory.shape}")
    print(trajectory[0])
    plot_trajectory_equal_axes(trajectory)
    pass
"""