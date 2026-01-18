import kiss_matcher
import numpy as np

from mcap_ros2.decoder import DecoderFactory

from mcap.reader import make_reader

resolution = 0.1

def remove_nan_from_point_cloud(point_cloud):
    return point_cloud[np.isfinite(point_cloud).any(axis=1)]

def registrate_point_clouds(src, tgt):
    src = remove_nan_from_point_cloud(src)
    tgt = remove_nan_from_point_cloud(tgt)
    params = kiss_matcher.KISSMatcherConfig(resolution)
    matcher = kiss_matcher.KISSMatcher(params)
    result = matcher.estimate(src, tgt)
    matcher.print()

    num_rot_inliers = matcher.get_num_rotation_inliers()
    num_final_inliers = matcher.get_num_final_inliers()
    # NOTE(hlim): By checking the final inliers, we can determine whether
    # the registration was successful or not. The larger the threshold,
    # the more conservatively the decision is made.
    # See https://github.com/MIT-SPARK/KISS-Matcher/issues/24
    thres_num_inliers = 5
    if (num_final_inliers < thres_num_inliers):
        print("\033[1;33m=> Registration might have failed :(\033[0m\n")
    else:
        print("\033[1;32m=> Registration likely succeeded XD\033[0m\n")

    # ------------------------------------------------------------
    # Visualization with Viser
    # ------------------------------------------------------------
    # Apply transformation to src
    rotation_matrix = np.array(result.rotation)
    translation_vector = np.array(result.translation)
    return rotation_matrix, translation_vector


import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import struct


def bag2np(bag_file, topic, is_repetitive=True):
    """
    Read a ROS2 bag file and extract PointCloud2 messages as numpy arrays.

    Args:
        bag_file: Path to the .mcap file
        topic: Topic name to read PointCloud2 messages from

    Returns:
        List of numpy arrays, one per PointCloud2 message
    """
    pointclouds = []
    decoder_factory = DecoderFactory()

    with open(bag_file, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        i = 0
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=[topic]):
            #print(f"{channel.topic} {schema.name} [{message.log_time}]: {ros_msg}")

            # Decode the message


            # Parse PointCloud2 data
            points = parse_pointcloud2(ros_msg)
            pointclouds.append(points)
            i += 1
            print(i)
            if is_repetitive or i == 10:
                return points

    return points


def parse_pointcloud2(msg):
    """
    Parse PointCloud2 message and extract XYZ points.
    Returns numpy array of shape (N, 3) with columns [x, y, z].

    The data field is a 1D byte array where points are packed according to:
    - point_step: bytes per point
    - row_step: bytes per row (point_step * width)
    - fields: describes the layout (offset, datatype) of each component
    """
    # Find x, y, z fields
    field_map = {}
    for field in msg.fields:
        field_map[field.name] = {
            'offset': field.offset,
            'datatype': field.datatype
        }

    # Datatype mapping (from sensor_msgs/PointField)
    dtype_map = {
        1: ('b', 1),  # INT8
        2: ('B', 1),  # UINT8
        3: ('h', 2),  # INT16
        4: ('H', 2),  # UINT16
        5: ('i', 4),  # INT32
        6: ('I', 4),  # UINT32
        7: ('f', 4),  # FLOAT32
        8: ('d', 8),  # FLOAT64
    }

    # Check if x, y, z exist
    if 'x' not in field_map or 'y' not in field_map or 'z' not in field_map:
        return np.array([]).reshape(0, 3)  # Empty array with shape (0, 3)

    # Get format for x, y, z
    x_info = field_map['x']
    y_info = field_map['y']
    z_info = field_map['z']

    # Assume same datatype for x, y, z (typically FLOAT32)
    fmt_char, size = dtype_map[x_info['datatype']]

    # Endianness
    endian_char = '>' if msg.is_bigendian else '<'

    # Extract points
    points = np.array([], dtype=np.float32).reshape(0, 3)
    total_written = 0
    num_points = msg.width * msg.height
    data = msg.data

    for i in range(num_points):
        offset = i * msg.point_step

        # Extract x, y, z
        x_offset = offset + x_info['offset']
        y_offset = offset + y_info['offset']
        z_offset = offset + z_info['offset']

        x = struct.unpack(endian_char + fmt_char,
                          bytes(data[x_offset:x_offset + size]))[0]
        y = struct.unpack(endian_char + fmt_char,
                          bytes(data[y_offset:y_offset + size]))[0]
        z = struct.unpack(endian_char + fmt_char,
                          bytes(data[z_offset:z_offset + size]))[0]

        # Skip NaN points if not dense
        if not msg.is_dense:
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
        new_row = np.array([x, y, z])
        points = np.append(points, [new_row], axis=0)
        #print(type(points))

    return points

    # Return as numpy array with shape (N, 3)
    if points:
        return np.array(points, dtype=np.float32)
    else:
        return np.array([]).reshape(0, 3)


import numpy as np

from mcap.reader import make_reader
from scipy.spatial.transform import Rotation as R


def get_initial_transform_mcap(bag_path, pose_src_topic, pose_tgt_topic):
    """
    Reads the first message from two topics in an MCAP bag and
    calculates the transform (Rotation/Translation) from src to tgt.
    """
    poses = {pose_src_topic: None, pose_tgt_topic: None}

    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        # Iterate through messages
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            topic = channel.topic
            #print(topic)

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

    print(type(p_s))
    print(p_t)

    # Convert positions to numpy vectors
    pos_s = np.array([p_s.position.x, p_s.position.y, p_s.position.z])
    pos_t = np.array([p_t.pose.position.x, p_t.pose.position.y, p_t.pose.position.z])

    # Convert orientations to SciPy Rotation objects
    # Note: ROS quaternions are [x, y, z, w], which SciPy accepts
    quat_s = [p_s.orientation.x, p_s.orientation.y, p_s.orientation.z, p_s.orientation.w]
    quat_t = [p_t.pose.orientation.x, p_t.pose.orientation.y, p_t.pose.orientation.z, p_t.pose.orientation.w]

    r_s = R.from_quat(quat_s)
    r_t = R.from_quat(quat_t)

    # 1. Rotation from src to tgt: R_rel = R_t * inv(R_s)
    rel_rot = r_t * r_s.inv()

    # 2. Translation from src to tgt in the world frame: t_rel = t_t - t_s
    rel_trans = pos_t - pos_s

    return rel_trans, rel_rot.as_matrix()


from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import io


def get_tf_tree_from_rosbag(bag_path):
    """
    Read a ROS2 bag (mcap format) and extract the TF tree at the start.

    Args:
        bag_path: Path to the .mcap file

    Returns:
        dict: TF tree as a nested dictionary where keys are parent frames
              and values are dicts of child frames with their transforms
              Format: {parent_frame: {child_frame: transform_data}}
    """
    tf_tree = {}
    tf_static_tree = {}

    with open(bag_path, 'rb') as f:
        reader = make_reader(f)
        decoder_factory = DecoderFactory()

        # Find tf and tf_static topics
        tf_topic_id = None
        tf_static_topic_id = None

        for channel_id, channel in reader.get_summary().channels.items():
            if channel.topic == '/tf':
                tf_topic_id = channel_id
            elif channel.topic == '/tf_static':
                tf_static_topic_id = channel_id

        # Read messages from the start of the bag
        for schema, channel, message in reader.iter_messages():
            # Only process tf and tf_static messages
            if channel.id not in [tf_topic_id, tf_static_topic_id]:
                continue

            # Decode the message
            decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
            tf_msg = decoder(message.data)

            # Extract transforms
            for transform in tf_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id

                transform_data = {
                    'translation': {
                        'x': transform.transform.translation.x,
                        'y': transform.transform.translation.y,
                        'z': transform.transform.translation.z
                    },
                    'rotation': {
                        'x': transform.transform.rotation.x,
                        'y': transform.transform.rotation.y,
                        'z': transform.transform.rotation.z,
                        'w': transform.transform.rotation.w
                    },
                    'timestamp': transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                }

                # Store in appropriate tree
                target_tree = tf_static_tree if channel.id == tf_static_topic_id else tf_tree

                if parent not in target_tree:
                    target_tree[parent] = {}
                target_tree[parent][child] = transform_data

            # Stop after reading initial messages
            if len(tf_tree) > 0 or len(tf_static_tree) > 0:
                break

    # Merge tf_static into tf_tree (static transforms take precedence)
    for parent, children in tf_static_tree.items():
        if parent not in tf_tree:
            tf_tree[parent] = {}
        tf_tree[parent].update(children)

    return tf_tree
# Example call:
# translation, rotation_matrix = get_initial_transform_mcap("my_data.mcap", "/gnss/pose", "/lidar/pose")

#print(get_tf_tree_from_rosbag("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap"))

"""



src = bag2np("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap", "/ec_swift/back_lidar_sensor/points_raw", is_repetitive=False)
tgt = bag2np("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap", "/sensor_station/lidar/points", is_repetitive=True)
#src = convert_for_kiss_matcher(src)
#tgt = convert_for_kiss_matcher(tgt)
print(src.shape)
print(tgt.shape)
sol = registrate_point_clouds(src, tgt)
print(sol[0])
print(sol[1])
gt_trans, gt_rot = get_initial_transform_mcap("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap", "/qualisys/TUDA_swift/pose", "/qualisys/TUDA_ouster/odom")
print(gt_trans)
print(gt_rot)
"""