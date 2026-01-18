from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2.reader import read_ros2_messages
import io
import struct


def copy_tf_subtree_to_bag(input_bag_path, output_bag_path, root_frame):
    """
    Copies a TF subtree starting from a given frame from input bag to output bag.
    Retains all original data from the output bag and appends filtered TF messages.

    Args:
        input_bag_path: Path to the source bag file containing TF data
        output_bag_path: Path to the destination bag file (will be read and rewritten)
        root_frame: The frame name to start copying from (includes this frame and all descendants)
    """

    import tempfile
    import shutil

    # Step 1: Read existing output bag - store raw MCAP records
    existing_records = []

    try:
        with open(output_bag_path, "rb") as f:
            reader = make_reader(f)

            # Read all messages as raw records
            for schema, channel, message in reader.iter_messages():
                existing_records.append({
                    'schema': schema,
                    'channel': channel,
                    'message': message
                })

        print(f"Read {len(existing_records)} existing messages from output bag")
    except FileNotFoundError:
        print(f"Output bag not found, will create new file: {output_bag_path}")

    # Step 2: Read and filter TF messages from input bag
    tf_messages = []

    with open(input_bag_path, "rb") as f:
        for msg in read_ros2_messages(f):
            # Check if this is a TF or TF_static message
            if msg.channel.topic in ["/tf", "/tf_static"]:
                # Parse the transform message
                transforms = parse_tf_message(msg.ros_msg)

                # Filter transforms that belong to the subtree
                filtered_transforms = filter_subtree_transforms(transforms, root_frame)

                if filtered_transforms:
                    # Store the full message info for later
                    tf_messages.append({
                        'msg': msg,
                        'filtered_transforms': filtered_transforms
                    })

    print(f"Found {len(tf_messages)} TF messages containing subtree transforms")

    # Step 3: Write everything to a temporary file first, then move it
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mcap')

    try:
        with open(temp_path, "wb") as f:
            writer = Writer(f)
            writer.start()

            # Track schemas and channels we've registered
            schema_ids = {}
            channel_ids = {}

            total_written = 0

            # Collect all messages with their metadata
            all_messages = []

            # Add existing messages
            for record in existing_records:
                timestamp = record['message'].log_time
                # Convert datetime to nanoseconds if needed
                if hasattr(timestamp, 'timestamp'):
                    timestamp = int(timestamp.timestamp() * 1e9)
                all_messages.append((
                    timestamp,
                    record['message'].data,
                    record['schema'],
                    record['channel']
                ))

            # Add filtered TF messages
            for tf_data in tf_messages:
                msg = tf_data['msg']
                filtered_transforms = tf_data['filtered_transforms']

                # Create new message with filtered transforms
                new_msg = reconstruct_tf_message(filtered_transforms, msg.ros_msg)

                # Serialize the message using CDR serialization
                serialized = serialize_ros2_message(new_msg)

                timestamp = msg.log_time
                # Convert datetime to nanoseconds if needed
                if hasattr(timestamp, 'timestamp'):
                    timestamp = int(timestamp.timestamp() * 1e9)

                all_messages.append((
                    timestamp,
                    serialized,
                    msg.schema,
                    msg.channel
                ))

            # Sort all messages by timestamp
            all_messages.sort(key=lambda x: x[0])

            # Write messages
            for timestamp, data, schema, channel in all_messages:
                # Register schema if needed
                schema_key = (schema.name, schema.encoding, schema.data)
                if schema_key not in schema_ids:
                    schema_id = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )
                    schema_ids[schema_key] = schema_id
                else:
                    schema_id = schema_ids[schema_key]

                # Register channel if needed
                channel_key = (channel.topic, channel.message_encoding, schema_id)
                if channel_key not in channel_ids:
                    channel_id = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_id,
                        metadata=channel.metadata
                    )
                    channel_ids[channel_key] = channel_id
                else:
                    channel_id = channel_ids[channel_key]

                # Write message
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp,
                    data=data,
                    publish_time=timestamp
                )
                total_written += 1

            writer.finish()

        # Close the temp file descriptor
        import os
        os.close(temp_fd)

        # Move temp file to output path
        shutil.move(temp_path, output_bag_path)

        print(f"Successfully wrote {total_written} messages to {output_bag_path}")

    except Exception as e:
        # Clean up temp file on error
        import os
        try:
            os.close(temp_fd)
            os.remove(temp_path)
        except:
            pass
        raise e


def parse_tf_message(ros_msg):
    """
    Parse a TF message to extract transform information.
    Returns list of (parent_frame, child_frame, transform_data) tuples.
    """
    transforms = []

    # TF messages contain a 'transforms' array
    # Each transform has header.frame_id (parent) and child_frame_id (child)
    if hasattr(ros_msg, 'transforms'):
        for transform in ros_msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id
            transforms.append((parent, child, transform))

    return transforms


def filter_subtree_transforms(transforms, root_frame):
    """
    Filter transforms to only include those in the subtree starting from root_frame.
    Uses iterative approach to find all descendants.
    """
    if not transforms:
        return []

    # Build frame relationship map
    children_map = {}
    transform_map = {}

    for parent, child, transform in transforms:
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(child)
        transform_map[(parent, child)] = transform

    # Find all frames in subtree starting from root_frame
    frames_in_subtree = {root_frame}
    to_process = [root_frame]

    while to_process:
        current = to_process.pop(0)
        if current in children_map:
            for child in children_map[current]:
                if child not in frames_in_subtree:
                    frames_in_subtree.add(child)
                    to_process.append(child)

    # Filter transforms - include if parent is in subtree
    # (child will be in subtree too if parent is)
    filtered = []
    for parent, child, transform in transforms:
        if parent in frames_in_subtree:
            filtered.append(transform)

    return filtered


def reconstruct_tf_message(filtered_transforms, original_msg):
    """
    Reconstruct a TF message with filtered transforms.
    Returns a new message object with the filtered transforms.
    """
    # Create a new message of the same type
    new_msg_type = type(original_msg)
    new_msg = new_msg_type()

    # Copy the filtered transforms
    new_msg.transforms = filtered_transforms

    return new_msg


def serialize_ros2_message(msg):
    """
    Serialize a ROS2 message to CDR format without using rclpy.
    """
    from io import BytesIO
    import struct

    # Simple CDR serialization for TFMessage
    # This is a basic implementation - may need adjustments for complex types
    buffer = BytesIO()

    # CDR header (encapsulation kind and options)
    buffer.write(struct.pack('<BxH', 0, 0))

    # Serialize the transforms array
    # First write the sequence length
    buffer.write(struct.pack('<I', len(msg.transforms)))

    # Then serialize each transform
    for transform in msg.transforms:
        # Serialize header
        serialize_header(buffer, transform.header)

        # Serialize child_frame_id (string)
        serialize_string(buffer, transform.child_frame_id)

        # Serialize transform (translation + rotation)
        serialize_vector3(buffer, transform.transform.translation)
        serialize_quaternion(buffer, transform.transform.rotation)

    return buffer.getvalue()


def serialize_header(buffer, header):
    """Serialize a std_msgs/Header"""
    import struct
    # Timestamp
    buffer.write(struct.pack('<ii', header.stamp.sec, header.stamp.nanosec))
    # frame_id (string)
    serialize_string(buffer, header.frame_id)


def serialize_string(buffer, s):
    """Serialize a string in CDR format"""
    import struct
    encoded = s.encode('utf-8')
    buffer.write(struct.pack('<I', len(encoded) + 1))  # +1 for null terminator
    buffer.write(encoded)
    buffer.write(b'\x00')  # null terminator
    # Add padding to align to 4 bytes
    padding = (4 - ((len(encoded) + 1) % 4)) % 4
    buffer.write(b'\x00' * padding)


def serialize_vector3(buffer, vec):
    """Serialize a geometry_msgs/Vector3"""
    import struct
    buffer.write(struct.pack('<ddd', vec.x, vec.y, vec.z))


def serialize_quaternion(buffer, quat):
    """Serialize a geometry_msgs/Quaternion"""
    import struct
    buffer.write(struct.pack('<dddd', quat.x, quat.y, quat.z, quat.w))


# Usage example:
if __name__ == "__main__":
    copy_tf_subtree_to_bag(
        input_bag_path='/home/aaron/hector/rosbag2_2026_01_15-14_53_10/rosbag2_2026_01_15-14_53_10_0.mcap',
        output_bag_path="/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap" ,
        root_frame="ec_swift/base_link"
    )