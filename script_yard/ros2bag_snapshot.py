# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


from pathlib import Path
import struct

from pathlib import Path


def read_rosbag2_topics(bag_path):
    """
    Read a ROS2 bag in MCAP format and print all topics.

    Args:
        bag_path: Path to the rosbag2 directory or .mcap file
    """
    try:
        from mcap.reader import make_reader
    except ImportError:
        print("Error: mcap library not installed. Install it with: pip install mcap")
        return

    bag_path = Path(bag_path)

    # Find the MCAP file
    if bag_path.is_dir():
        mcap_files = list(bag_path.glob("*.mcap"))
        if not mcap_files:
            print(f"No .mcap file found in {bag_path}")
            return
        mcap_file = mcap_files[0]
    else:
        mcap_file = bag_path

    # Read the MCAP file
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)

        # Get summary information
        summary = reader.get_summary()

        if summary is None:
            print("No summary available in MCAP file")
            return

        # Print topics
        print(f"Found {len(summary.channels)} topics in bag:\n")
        print(f"{'Channel ID':<12} {'Topic Name':<50} {'Message Type':<30} {'Schema'}")
        print("-" * 130)

        # Track message counts
        topic_counts = {}

        for channel_id, channel in summary.channels.items():
            schema_name = ""
            if channel.schema_id in summary.schemas:
                schema_name = summary.schemas[channel.schema_id].name

            print(f"{channel_id:<12} {channel.topic:<50} {channel.message_encoding:<30} {schema_name}")
            topic_counts[channel.topic] = 0

        # Count messages from statistics
        print("\n\nMessage counts:")
        print(f"{'Topic Name':<50} {'Count'}")
        print("-" * 70)

        if summary.statistics:
            # statistics is a Statistics object, not a dict
            if hasattr(summary.statistics, 'channel_message_counts'):
                for channel_id, count in summary.statistics.channel_message_counts.items():
                    if channel_id in summary.channels:
                        topic_name = summary.channels[channel_id].topic
                        topic_counts[topic_name] = count

        # If statistics didn't work, count manually by iterating messages
        if all(count == 0 for count in topic_counts.values()):
            print("Counting messages manually (this may take a moment)...")
            for schema, channel, message in reader.iter_messages():
                if channel.topic in topic_counts:
                    topic_counts[channel.topic] += 1

        # Print sorted by count
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        for topic, count in sorted_topics:
            print(f"{topic:<50} {count}")

        # Print overall statistics
        if summary.statistics:
            print(f"\n\nTotal messages: {summary.statistics.message_count}")
            print(f"Start time: {summary.statistics.message_start_time}")
            print(f"End time: {summary.statistics.message_end_time}")
            duration_sec = (summary.statistics.message_end_time - summary.statistics.message_start_time) / 1e9
            print(f"Duration: {duration_sec:.2f} seconds")


def print_topic(bag_path, topic):
    """
    Print all messages from a specific topic in an MCAP rosbag.

    Args:
        bag_path: Path to the .mcap file
        topic: Name of the topic to print
    """
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        print("Error: Required libraries not installed.")
        print("Install with: pip install mcap mcap-ros2-support")
        return

    from pathlib import Path

    bag_path = Path(bag_path)

    if not bag_path.exists():
        print(f"File not found: {bag_path}")
        return

    message_count = 0
    decoder_factory = DecoderFactory()

    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[decoder_factory])

        print(f"Reading topic: {topic}\n")
        print("=" * 80)

        # First, get the summary to find matching channels
        summary = reader.get_summary()
        if not summary:
            print("No summary available in MCAP file")
            return

        # Find channels that match the topic
        matching_channels = [ch for ch in summary.channels.values() if ch.topic == topic]

        if not matching_channels:
            print(f"No channels found for topic: {topic}")
            print("\nAvailable topics:")
            for channel in summary.channels.values():
                print(f"  - {channel.topic}")
            return

        # Iterate through all messages
        for msg in reader.iter_messages():
            schema, channel, message = msg[:3]

            if channel.topic == topic:
                message_count += 1
                timestamp_sec = message.log_time / 1e9

                print(f"\nMessage #{message_count}")
                print(f"Timestamp: {timestamp_sec:.6f} seconds")
                print(f"Topic: {channel.topic}")
                print(f"Schema: {schema.name}")
                print("-" * 80)

                # Try to decode the message
                try:
                    decoded = decoder_factory.decoder_for(
                        message_encoding=channel.message_encoding,
                        schema=schema
                    ).decode(message.data)
                    print(decoded)
                except Exception as e:
                    print(f"Could not decode message: {e}")
                    print(f"Raw data size: {len(message.data)} bytes")
                    if len(message.data) <= 200:
                        print(f"Raw data: {message.data}")

                print("=" * 80)

        if message_count == 0:
            print(f"No messages found for topic: {topic}")
        else:
            print(f"\nTotal messages printed: {message_count}")


def cut_out(bag_folder, start_time_stamp, end_time_stamp):
    """
    Extract all topic messages within a time interval from multiple MCAP bag files.

    Args:
        bag_folder: Path to folder containing MCAP files
        start_time_stamp: Start timestamp in nanoseconds
        end_time_stamp: End timestamp in nanoseconds

    Returns:
        Dictionary with structure: {topic_name: [(timestamp, message_data, schema, channel), ...]}
    """
    try:
        from mcap.reader import make_reader
    except ImportError:
        print("Error: mcap library not installed. Install it with: pip install mcap")
        return {}

    from pathlib import Path
    import re

    bag_folder = Path(bag_folder)

    if not bag_folder.exists():
        print(f"Folder not found: {bag_folder}")
        return {}

    # Find all MCAP files in the folder
    all_mcap_files = list(bag_folder.glob("*.mcap"))

    if not all_mcap_files:
        print(f"No MCAP files found in {bag_folder}")
        return {}

    # Extract file numbers and sort
    # Pattern: rosbag2_YYYY_MM_DD-HH_MM_SS_cropped_N.mcap
    file_info = []
    pattern = r'_(\d+)\.mcap$'

    for mcap_file in all_mcap_files:
        match = re.search(pattern, mcap_file.name)
        if match:
            file_number = int(match.group(1))
            file_info.append((file_number, mcap_file))
        else:
            # Handle files without numbers (might be the first one without _0)
            # Check if it ends with .mcap but has no number before it
            if mcap_file.name.endswith('.mcap') and not re.search(r'_\d+\.mcap$', mcap_file.name):
                file_info.append((0, mcap_file))

    # Sort by file number
    file_info.sort(key=lambda x: x[0])
    mcap_files = [f[1] for f in file_info]

    if not mcap_files:
        print(f"No properly numbered MCAP files found in {bag_folder}")
        return {}

    print(f"Found {len(mcap_files)} bag files:")
    for num, mcap_file in file_info:
        print(f"  [{num}] {mcap_file.name}")

    # Dictionary to store messages by topic
    topic_messages = {}
    total_messages = 0

    # Process each bag file
    for bag_file in mcap_files:
        print(f"\nProcessing {bag_file.name}...")

        with open(bag_file, "rb") as f:
            reader = make_reader(f)

            # Get summary to check time range
            summary = reader.get_summary()
            if summary and summary.statistics:
                bag_start = summary.statistics.message_start_time
                bag_end = summary.statistics.message_end_time

                print(f"  Bag time range: {bag_start / 1e9:.6f} to {bag_end / 1e9:.6f} seconds")

                # Skip bag if it's entirely outside our time range
                if bag_end < start_time_stamp or bag_start > end_time_stamp:
                    print(f"  Skipping {bag_file.name} (outside time range)")
                    continue

            # Iterate through messages
            messages_in_file = 0
            for schema, channel, message in reader.iter_messages():
                timestamp = message.log_time

                # Check if message is within time range
                if start_time_stamp <= timestamp <= end_time_stamp:
                    topic_name = channel.topic

                    # Initialize topic list if needed
                    if topic_name not in topic_messages:
                        topic_messages[topic_name] = []

                    # Store timestamp, message data, schema, and channel
                    topic_messages[topic_name].append((timestamp, message.data, schema, channel))
                    total_messages += 1
                    messages_in_file += 1

            print(f"  Extracted {messages_in_file} messages from this file")

    # Sort messages by timestamp within each topic
    for topic in topic_messages:
        topic_messages[topic].sort(key=lambda x: x[0])

    print(f"\n{'=' * 80}")
    print(f"Extraction complete!")
    print(f"Total messages extracted: {total_messages}")
    print(f"Number of topics: {len(topic_messages)}")
    print(f"Time range: {start_time_stamp / 1e9:.6f} to {end_time_stamp / 1e9:.6f} seconds")

    # Print summary
    print(f"\nMessages per topic:")
    for topic, messages in sorted(topic_messages.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {topic}: {len(messages)} messages")

    return topic_messages


# Helper function to write extracted messages to a new MCAP file
def save_cut_out(topic_messages, output_file):
    """
    Save extracted messages to a new MCAP file.

    Args:
        topic_messages: Dictionary returned by cut_out()
        output_file: Path to output MCAP file
    """
    try:
        from mcap.writer import Writer
    except ImportError:
        print("Error: mcap library not installed")
        return

    from pathlib import Path

    output_file = Path(output_file)

    print(f"\nWriting to {output_file}...")

    with open(output_file, "wb") as f:
        writer = Writer(f)
        writer.start()

        # Track schemas and channels we've registered
        schema_ids = {}
        channel_ids = {}

        total_written = 0

        # Write all messages sorted by timestamp across all topics
        all_messages = []
        for topic, messages in topic_messages.items():
            all_messages.extend(messages)

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

    print(f"Successfully wrote {total_written} messages to {output_file}")



#save_cut_out(messages, "/home/aaron/output_cropped.mcap")

# Or access messages programmatically
# for topic, msgs in messages.items():
#     print(f"\nTopic: {topic}")
#     for timestamp, data, schema, channel in msgs[:5]:  # Print first 5
#         print(f"  Timestamp: {timestamp / 1e9:.6f}")
# Example usage:
# print_topic("/home/aaron/uni/thesis/data/seq4/seq4_swift/rosbag2_2025_07_29-14_50_52_cropped_1.mcap", "/ec_swift/back_lidar_sensor/imu")

# Example usage:
# print_topic("/home/aaron/uni/thesis/data/seq4/seq4_swift/", "/ec_swift/back_lidar_sensor/imu")
# Example usage:
# print_topic("/home/aaron/uni/thesis/data/seq4/seq4_swift/rosbag2_2025_07_29-14_50_52_cropped_1.mcap", "/ec_swift/back_lidar_sensor/imu")
# Example usage:
# print_topic("/home/aaron/uni/thesis/data/seq4/seq4_swift/rosbag2_2025_07_29-14_50_52_cropped_1.mcap", "/ec_swift/imu/data")

# Example usage:
# read_rosbag2_topics("/path/to/your/rosbag2_folder")
# or
# read_rosbag2_topics("/path/to/your/bag.mcap")

# Example usage:
# read_rosbag2_topics("/path/to/your/rosbag2_folder")
# or
# read_rosbag2_topics("/path/to/your/bag.mcap")
path_swift = "/home/aaron/uni/thesis/data/seq4/seq4_swift/"
path_telemax = "/home/aaron/uni/thesis/data/seq4/seq4_telemax/"
path_mocap = "/home/aaron/uni/thesis/data/seq4/seq4_mocap/"
path_ouster = "/home/aaron/uni/thesis/data/seq4/seq4_ouster/"
# Press the green button in the gutter to run the script.
timestamps_swift_telemax = [[1753793572896000000, 1753793575124138328], [1753793557963907881 ,1753793563153451185],]

relationship = [["swift", "telemax"], ["swift", "telemax"], ]

from mcap_ros2.reader import read_ros2_messages
from mcap_ros2.writer import Writer as McapWriter
from mcap.writer import Writer as McapCoreWriter
import io

from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter
import struct
from io import BytesIO


def extract_and_copy_tf_tree(input_bag_path, output_bag_path, root_frame="ec_swift/base_link"):
    """
    Extract all TF transforms that are descendants of root_frame from input bag
    and write them to output bag.

    Args:
        input_bag_path: Path to input MCAP file
        output_bag_path: Path to output MCAP file
        root_frame: The root frame to start extraction from (default: "ec_swift/base_link")
    """

    # Read input bag and collect all TF data
    tf_frames = set()
    tf_messages = []

    print(f"Reading TF data from {input_bag_path}...")

    with open(input_bag_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        # Get schemas and channels
        schemas = {schema.id: schema for schema in summary.schemas.values()}
        channels = {channel.id: channel for channel in summary.channels.values()}

        for schema_id, channel_id, message in reader.iter_messages():
            channel = channels[channel_id]

            if channel.topic in ["/tf", "/tf_static"]:
                schema = schemas[channel.schema_id]

                # Parse CDR data
                transforms = parse_tf_message(message.data)
                tf_messages.append((channel.topic, message.log_time, message.data, schema, transforms))

                # Collect all frames
                for transform in transforms:
                    tf_frames.add(transform['parent_frame'])
                    tf_frames.add(transform['child_frame'])

    print(f"Found {len(tf_frames)} unique frames")

    # Build parent-child relationships
    parent_child_map = {}
    for topic, timestamp, data, schema, transforms in tf_messages:
        for transform in transforms:
            parent = transform['parent_frame']
            child = transform['child_frame']
            if child not in parent_child_map:
                parent_child_map[child] = parent

    # Find all descendants of root_frame
    def get_descendants(frame):
        descendants = {frame}
        for child, parent in parent_child_map.items():
            if parent == frame:
                descendants.update(get_descendants(child))
        return descendants

    relevant_frames = get_descendants(root_frame)
    print(f"Found {len(relevant_frames)} frames descended from {root_frame}")
    print(f"Frames: {sorted(relevant_frames)}")

    # Write filtered TF data to output bag
    print(f"\nWriting filtered TF data to {output_bag_path}...")

    with open(output_bag_path, "wb") as f:
        writer = McapWriter(f)
        writer.start()

        # Track registered schemas and channels
        schema_ids = {}
        channel_ids = {}

        for topic, timestamp, data, schema, transforms in tf_messages:
            # Filter transforms to only include relevant frames
            filtered_transforms = [
                t for t in transforms
                if t['parent_frame'] in relevant_frames and t['child_frame'] in relevant_frames
            ]

            if filtered_transforms:
                # Serialize filtered message
                filtered_data = serialize_tf_message(filtered_transforms)

                # Register schema if needed
                if schema.name not in schema_ids:
                    schema_id = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )
                    schema_ids[schema.name] = schema_id
                else:
                    schema_id = schema_ids[schema.name]

                # Register channel if needed
                channel_key = (topic, schema.name)
                if channel_key not in channel_ids:
                    channel_id = writer.register_channel(
                        topic=topic,
                        message_encoding="cdr",
                        schema_id=schema_id
                    )
                    channel_ids[channel_key] = channel_id
                else:
                    channel_id = channel_ids[channel_key]

                # Write message
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp,
                    data=filtered_data,
                    publish_time=timestamp
                )

        writer.finish()

    print(f"Done! Output written to {output_bag_path}")


def parse_tf_message(data):
    """Parse CDR-encoded TFMessage"""
    buffer = BytesIO(data)
    transforms = []

    # Skip CDR encapsulation header (4 bytes)
    buffer.read(4)

    # Read number of transforms
    num_transforms = struct.unpack('<I', buffer.read(4))[0]

    for _ in range(num_transforms):
        transform = {}

        # Header - timestamp
        sec = struct.unpack('<i', buffer.read(4))[0]
        nanosec = struct.unpack('<I', buffer.read(4))[0]
        transform['timestamp'] = (sec, nanosec)

        # Header - frame_id (parent frame)
        frame_id_len = struct.unpack('<I', buffer.read(4))[0]
        frame_id = buffer.read(frame_id_len).rstrip(b'\x00').decode('utf-8')
        transform['parent_frame'] = frame_id

        # Align to 4 bytes
        pos = buffer.tell()
        if pos % 4 != 0:
            buffer.read(4 - pos % 4)

        # child_frame_id
        child_frame_id_len = struct.unpack('<I', buffer.read(4))[0]
        child_frame_id = buffer.read(child_frame_id_len).rstrip(b'\x00').decode('utf-8')
        transform['child_frame'] = child_frame_id

        # Align to 8 bytes for doubles
        pos = buffer.tell()
        if pos % 8 != 0:
            buffer.read(8 - pos % 8)

        # Transform - translation
        translation_x = struct.unpack('<d', buffer.read(8))[0]
        translation_y = struct.unpack('<d', buffer.read(8))[0]
        translation_z = struct.unpack('<d', buffer.read(8))[0]
        transform['translation'] = (translation_x, translation_y, translation_z)

        # Transform - rotation (quaternion)
        rotation_x = struct.unpack('<d', buffer.read(8))[0]
        rotation_y = struct.unpack('<d', buffer.read(8))[0]
        rotation_z = struct.unpack('<d', buffer.read(8))[0]
        rotation_w = struct.unpack('<d', buffer.read(8))[0]
        transform['rotation'] = (rotation_x, rotation_y, rotation_z, rotation_w)

        transforms.append(transform)

    return transforms


def serialize_tf_message(transforms):
    """Serialize transforms to CDR-encoded TFMessage"""
    buffer = BytesIO()

    # CDR encapsulation header (little endian)
    buffer.write(b'\x00\x01\x00\x00')

    # Number of transforms
    buffer.write(struct.pack('<I', len(transforms)))

    for transform in transforms:
        # Header - timestamp
        sec, nanosec = transform['timestamp']
        buffer.write(struct.pack('<i', sec))
        buffer.write(struct.pack('<I', nanosec))

        # Header - frame_id (parent frame)
        frame_id_bytes = transform['parent_frame'].encode('utf-8')
        buffer.write(struct.pack('<I', len(frame_id_bytes) + 1))
        buffer.write(frame_id_bytes)
        buffer.write(b'\x00')

        # Align to 4 bytes
        pos = buffer.tell()
        if pos % 4 != 0:
            buffer.write(b'\x00' * (4 - pos % 4))

        # child_frame_id
        child_frame_id_bytes = transform['child_frame'].encode('utf-8')
        buffer.write(struct.pack('<I', len(child_frame_id_bytes) + 1))
        buffer.write(child_frame_id_bytes)
        buffer.write(b'\x00')

        # Align to 8 bytes for doubles
        pos = buffer.tell()
        if pos % 8 != 0:
            buffer.write(b'\x00' * (8 - pos % 8))

        # Transform - translation
        tx, ty, tz = transform['translation']
        buffer.write(struct.pack('<d', tx))
        buffer.write(struct.pack('<d', ty))
        buffer.write(struct.pack('<d', tz))

        # Transform - rotation (quaternion)
        rx, ry, rz, rw = transform['rotation']
        buffer.write(struct.pack('<d', rx))
        buffer.write(struct.pack('<d', ry))
        buffer.write(struct.pack('<d', rz))
        buffer.write(struct.pack('<d', rw))

    return buffer.getvalue()


# Usage example:


def save_snapshots(timestamps, output_dir="/home/aaron/uni/thesis/data/snapshots"):
    """
    Save snapshot files for each pair of timestamps.

    Parameters:
    -----------
    timestamps : numpy.ndarray
        Array of shape (N, 2) containing [start_time, end_time] pairs
    output_dir : str
        Directory to save the snapshot files (default: "/home/aaron/uni/thesis/data/snapshots")
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(timestamps)
    for idx, (start_time, end_time) in enumerate(timestamps):
        print(start_time, end_time)
        # Cut out segments from each bag
        swift = cut_out(path_swift, start_time, end_time)
        ouster = cut_out(path_ouster, start_time, end_time)
        #telemax = cut_out(path_telemax, start_time, end_time)
        mocap = cut_out(path_mocap, start_time, end_time)

        # Merge bags
        swift |= ouster
        swift |= mocap
       # swift |= telemax

        # Create filename with timestamp and index
        duration = end_time - start_time
        filename = f"snapshot_{idx:03d}_t{start_time:.3f}_d{duration:.3f}s.mcap"
        output_path = os.path.join(output_dir, filename)

        # Save the merged bag
        save_cut_out(swift, output_path)

        print(f"Saved snapshot {idx + 1}/{len(timestamps)}: {filename}")

    print(f"\nAll {len(timestamps)} snapshots saved to {output_dir}")
from static_timestamps import extract_static_intervals
import numpy as np
if __name__ == '__main__':
    #extract_and_copy_tf_tree('/home/aaron/hector/rosbag2_2026_01_15-14_53_10/rosbag2_2026_01_15-14_53_10_0.mcap',"/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap" )

    save_snapshots((np.array([
    [1753793456188, 1753793458188],
    [1753793616609, 1753793618609],
    [1753793691949, 1753793693949],
    [1753793735150, 1753793737150],
    [1753793821618, 1753793823618],
    [1753793846517, 1753793848517],
    [1753793877328, 1753793879328]
])*1e6))
    #read_rosbag2_topics("/home/aaron/uni/thesis/data/seq4/seq4_swift/rosbag2_2025_07_29-14_50_52_cropped_1.mcap")
    #print_topic(path, "/ec_swift/back_lidar_sensor/imu ")
    #save_cut_out(cut_out(path, 1753793572896000000, 1753793575124138328), "/home/aaron/uni/thesis/data/snapshots/test1.mcap")
    #read_rosbag2_topics("/home/aaron/uni/thesis/data/snapshots/swift_telemax.mcap")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
