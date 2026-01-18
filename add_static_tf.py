import numpy as np
from pathlib import Path
from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter
from mcap.well_known import SchemaEncoding, MessageEncoding
import struct
import time


def add_static_transform_to_bag(input_bag_path, output_bag_path):
    """
    Read a ROS2 bag, add static transforms base_link -> back_lidar_sensor_frame
    and base_link -> front_lidar_sensor_frame, and save to a new bag.

    Args:
        input_bag_path: Path to input .mcap file
        output_bag_path: Path to output .mcap file
    """

    # Define the static transform matrices
    T_base_to_back_lidar = np.array([
        [np.sqrt(3) / 2, 0.0, -0.5, -0.19],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, np.sqrt(3) / 2, 0.2204],
        [0.0, 0.0, 0.0, 1.0]
    ])

    T_base_to_front_lidar = np.array([
        [-np.sqrt(3) / 2, 0.0, -0.5, 0.19],
        [0.0, -1.0, 0.0, 0.0],
        [0.5, 0.0, np.sqrt(3) / 2, 0.2204],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Extract translations and rotations (as quaternions)
    transforms = [
        {
            'parent': 'base_link',
            'child': 'back_lidar_sensor_frame',
            'translation': T_base_to_back_lidar[:3, 3],
            'quaternion': rotation_matrix_to_quaternion(T_base_to_back_lidar[:3, :3])
        },
        {
            'parent': 'base_link',
            'child': 'front_lidar_sensor_frame',
            'translation': T_base_to_front_lidar[:3, 3],
            'quaternion': rotation_matrix_to_quaternion(T_base_to_front_lidar[:3, :3])
        }
    ]

    print(f"Reading from: {input_bag_path}")
    for tf in transforms:
        print(f"\n{tf['parent']} -> {tf['child']}:")
        print(
            f"  Translation: x={tf['translation'][0]:.4f}, y={tf['translation'][1]:.4f}, z={tf['translation'][2]:.4f}")
        print(
            f"  Quaternion: x={tf['quaternion'][0]:.4f}, y={tf['quaternion'][1]:.4f}, z={tf['quaternion'][2]:.4f}, w={tf['quaternion'][3]:.4f}")

    # Create output directory
    output_path = Path(output_bag_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read input bag and write to output bag
    with open(input_bag_path, 'rb') as input_file:
        reader = make_reader(input_file)
        summary = reader.get_summary()

        with open(output_bag_path, 'wb') as output_file:
            writer = McapWriter(output_file)

            # Start with ROS2 profile metadata
            writer.start(profile="ros2", library="mcap python")

            # Track schemas and channels
            schema_map = {}
            channel_map = {}
            static_transform_added = False
            first_timestamp = None
            tf_static_channel_id = None

            # First pass: copy schemas and channels
            for schema, channel, message in reader.iter_messages():
                # Store schema
                if schema.id not in schema_map:
                    new_schema = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )
                    schema_map[schema.id] = new_schema

                # Store channel
                if channel.id not in channel_map:
                    # Copy channel metadata
                    channel_metadata = {}
                    if channel.metadata:
                        channel_metadata = dict(channel.metadata)

                    new_channel = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_map[schema.id],
                        metadata=channel_metadata
                    )
                    channel_map[channel.id] = new_channel

                # Track first timestamp
                if first_timestamp is None:
                    first_timestamp = message.log_time

            # Reset reader
            input_file.seek(0)
            reader = make_reader(input_file)

            # Create tf_static schema and channel
            tf_static_schema = create_tf_message_schema(writer)
            tf_static_channel_id = writer.register_channel(
                topic="/tf_static",
                message_encoding="cdr",
                schema_id=tf_static_schema,
                metadata={
                    "offered_qos_profiles": "- history: 3\n  depth: 0\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 2147483647\n    nsec: 4294967295\n  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 2147483647\n    nsec: 4294967295\n  avoid_ros_namespace_conventions: false\n"
                }
            )

            # Second pass: write messages
            for schema, channel, message in reader.iter_messages():
                # Write original message
                writer.add_message(
                    channel_id=channel_map[channel.id],
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time
                )

                # Add static transform after first message
                if not static_transform_added and first_timestamp is not None:
                    # Create TFMessage with both transforms
                    tf_data = create_tf_message_cdr_multiple(
                        first_timestamp,
                        transforms
                    )

                    writer.add_message(
                        channel_id=tf_static_channel_id,
                        log_time=first_timestamp,
                        data=tf_data,
                        publish_time=first_timestamp
                    )

                    static_transform_added = True
                    print(f"\nAdded static transforms to /tf_static")

            writer.finish()

    print(f"\nSaved to: {output_bag_path}")


def create_tf_message_schema(writer):
    """Create schema for TFMessage (tf2_msgs/msg/TFMessage) in ros2msg format"""

    # TFMessage schema in ROS2 msg format
    schema_text = """# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id at the time of header.stamp
#
# This message is mostly used by the 
# <a href="https://index.ros.org/p/tf2_ros/">tf2_ros</a> package.
# See its documentation for more information.
#
# The child_frame_id is necessary in addition to the frame_id
# in the Header to communicate the full reference for the transform
# in a self contained message.

geometry_msgs/TransformStamped[] transforms

================================================================================
MSG: geometry_msgs/TransformStamped
# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id at the time of header.stamp
#
# This message is mostly used by the 
# <a href="https://index.ros.org/p/tf2/">tf2</a> package.
# See its documentation for more information.

std_msgs/Header header
string child_frame_id # the frame id of the child frame
Transform transform

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id

================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Transform
# This represents the transform between two coordinate frames in free space.

Vector3 translation
Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.

# This is semantically different than a point.
# A vector is always anchored at the origin.
# When a transform is applied to a vector, only the rotational component is applied.

float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x 0
float64 y 0
float64 z 0
float64 w 1
"""

    # Register schema with ros2msg encoding
    schema_id = writer.register_schema(
        name="tf2_msgs/msg/TFMessage",
        encoding="ros2msg",
        data=schema_text.encode('utf-8')
    )

    return schema_id


def create_tf_message_cdr_multiple(timestamp_ns, transforms):
    """
    Create a CDR-encoded TFMessage containing multiple TransformStamped messages.

    Args:
        timestamp_ns: Timestamp in nanoseconds
        transforms: List of dicts with keys 'parent', 'child', 'translation', 'quaternion'

    Returns:
        bytes: CDR-encoded message
    """

    # CDR encapsulation header (little-endian, CDR v1)
    cdr_header = bytes([0x00, 0x01, 0x00, 0x00])

    # Build the message data
    data = bytearray()

    # Sequence length (number of transforms)
    data.extend(struct.pack('<I', len(transforms)))

    # Time components (same for all transforms)
    seconds = timestamp_ns // 1_000_000_000
    nanoseconds = timestamp_ns % 1_000_000_000

    # Add each transform
    for tf in transforms:
        parent_frame = tf['parent']
        child_frame = tf['child']
        translation = tf['translation']
        quaternion = tf['quaternion']

        # Header.stamp
        data.extend(struct.pack('<i', int(seconds)))
        data.extend(struct.pack('<I', int(nanoseconds)))

        # Header.frame_id (string: length + data)
        parent_frame_bytes = parent_frame.encode('utf-8')
        data.extend(struct.pack('<I', len(parent_frame_bytes) + 1))  # +1 for null terminator
        data.extend(parent_frame_bytes)
        data.append(0)  # null terminator

        # Align to 4-byte boundary
        while len(data) % 4 != 0:
            data.append(0)

        # child_frame_id (string: length + data)
        child_frame_bytes = child_frame.encode('utf-8')
        data.extend(struct.pack('<I', len(child_frame_bytes) + 1))
        data.extend(child_frame_bytes)
        data.append(0)

        # Align to 8-byte boundary for doubles
        while len(data) % 8 != 0:
            data.append(0)

        # Transform.translation (Vector3: x, y, z as doubles)
        data.extend(struct.pack('<ddd',
                                float(translation[0]),
                                float(translation[1]),
                                float(translation[2])))

        # Transform.rotation (Quaternion: x, y, z, w as doubles)
        data.extend(struct.pack('<dddd',
                                float(quaternion[0]),
                                float(quaternion[1]),
                                float(quaternion[2]),
                                float(quaternion[3])))

    return cdr_header + bytes(data)


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].

    Args:
        R: 3x3 numpy array representing rotation matrix

    Returns:
        Quaternion as numpy array [x, y, z, w]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w])


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Process all files in directory
    input_dir = Path("/home/aaron/uni/thesis/data/seq4/seq4_swift")
    output_dir = Path("/home/aaron/uni/thesis/data/seq4/seq4_swift_added_tfstatic")
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.mcap"):
        output_file = output_dir / input_file.name
        print(f"\n{'=' * 60}")
        print(f"Processing: {input_file.name}")
        print('=' * 60)
        add_static_transform_to_bag(str(input_file), str(output_file))