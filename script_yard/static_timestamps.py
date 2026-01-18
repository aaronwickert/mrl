import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def extract_static_intervals(bag_path, pose_topics, position_threshold=0.01, orientation_threshold=0.01,
                             static_duration=2.0):
    """
    Extract time intervals where all tracked poses remain static for at least 2 seconds.

    Parameters:
    -----------
    bag_path : str
        Path to the MCAP bag file
    pose_topics : list of str
        List of topic names containing pose messages (PoseStamped, PoseWithCovarianceStamped, etc.)
    position_threshold : float
        Maximum position change (meters) to consider static (default: 0.01m = 1cm)
    orientation_threshold : float
        Maximum orientation change (radians) to consider static (default: 0.01 rad)
    static_duration : float
        Required duration (seconds) of static behavior (default: 2.0s)

    Returns:
    --------
    numpy.ndarray
        Array of shape (N, 2) containing [start_time, end_time] in seconds for each static interval
    """

    # Storage for pose data: {topic: [(timestamp, pose), ...]}
    pose_data = {topic: [] for topic in pose_topics}

    # Read MCAP file
    decoder_factory = DecoderFactory()

    with open(bag_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[decoder_factory])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=pose_topics):
            timestamp = message.log_time / 1e9  # Convert nanoseconds to seconds

            # Extract pose from different message types
            if hasattr(ros_msg, 'pose'):
                if hasattr(ros_msg.pose, 'pose'):  # PoseWithCovarianceStamped
                    pose = ros_msg.pose.pose
                else:  # PoseStamped
                    pose = ros_msg.pose
            else:
                continue

            pose_data[channel.topic].append((timestamp, pose))

    # Sort all pose data by timestamp
    for topic in pose_data:
        pose_data[topic].sort(key=lambda x: x[0])

    # Check if we have data for all topics
    if not all(pose_data.values()):
        return np.array([]).reshape(0, 2)

    # Get all unique timestamps across all topics
    all_timestamps = sorted(set(ts for data in pose_data.values() for ts, _ in data))

    if len(all_timestamps) < 2:
        return np.array([]).reshape(0, 2)

    # For each topic, create interpolated pose lookup
    def get_pose_at_time(topic, timestamp):
        """Get the most recent pose for a topic at given timestamp"""
        data = pose_data[topic]
        for i in range(len(data) - 1, -1, -1):
            if data[i][0] <= timestamp:
                return data[i][1]
        return None

    # Track static intervals using sliding window
    static_intervals = []
    window_start_idx = 0

    for i, current_time in enumerate(all_timestamps):
        # Move window start forward to maintain 2-second window
        while window_start_idx < i and current_time - all_timestamps[window_start_idx] > static_duration:
            window_start_idx += 1

        window_start_time = all_timestamps[window_start_idx]
        window_duration = current_time - window_start_time

        # Check if we have at least 2 seconds of data
        if window_duration < static_duration:
            continue

        # Check if all poses are static throughout this window
        all_static = True

        for topic in pose_topics:
            # Get pose at start and end of window
            start_pose = get_pose_at_time(topic, window_start_time)
            end_pose = get_pose_at_time(topic, current_time)

            if start_pose is None or end_pose is None:
                all_static = False
                break

            # Calculate position change
            pos_change = np.sqrt(
                (end_pose.position.x - start_pose.position.x) ** 2 +
                (end_pose.position.y - start_pose.position.y) ** 2 +
                (end_pose.position.z - start_pose.position.z) ** 2
            )

            # Calculate orientation change (quaternion dot product)
            q1 = np.array([start_pose.orientation.x, start_pose.orientation.y,
                           start_pose.orientation.z, start_pose.orientation.w])
            q2 = np.array([end_pose.orientation.x, end_pose.orientation.y,
                           end_pose.orientation.z, end_pose.orientation.w])

            # Normalize quaternions
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)

            # Angular distance between quaternions
            dot_product = abs(np.dot(q1, q2))
            ori_change = 2 * np.arccos(np.clip(dot_product, 0, 1))

            # Check against thresholds
            if pos_change > position_threshold or ori_change > orientation_threshold:
                all_static = False
                break

        if all_static:
            # Check if this extends an existing interval or starts a new one
            if static_intervals and abs(static_intervals[-1][1] - window_start_time) < 0.001:
                # Extend the last interval
                static_intervals[-1][1] = current_time
            else:
                # Start new interval
                static_intervals.append([window_start_time, current_time])

    # Merge overlapping or adjacent intervals
    if static_intervals:
        merged = []
        current = static_intervals[0]

        for interval in static_intervals[1:]:
            if interval[0] <= current[1] + 0.001:  # Adjacent or overlapping
                current[1] = max(current[1], interval[1])
            else:
                merged.append(current)
                current = interval
        merged.append(current)

        return np.array(merged)

    return np.array([]).reshape(0, 2)

# Example usage:
