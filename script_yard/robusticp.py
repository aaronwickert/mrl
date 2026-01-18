import numpy as np
from mcap_ros2.decoder import DecoderFactory
import struct
from mcap.reader import make_reader
import subprocess


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


def save_point_cloud_ply(points, filename):
    """
    Save a point cloud to a .ply file.

    Parameters:
    -----------
    points : np.ndarray
        Point cloud array of shape (N, 3) containing XYZ coordinates
    filename : str
        Output filename (should end with .ply)
    """
    if not filename.endswith('.ply'):
        filename += '.ply'

    num_points = points.shape[0]

    # Create PLY header
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""

    # Write to file
    with open(filename, 'w') as f:
        f.write(header)
        # Write point coordinates
        np.savetxt(f, points, fmt='%.6f')

    print(f"Saved {num_points} points to {filename}")

#src = bag2np("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap", "/ec_swift/back_lidar_sensor/points_raw", is_repetitive=False)
#tgt = bag2np("/home/aaron/uni/thesis/data/snapshots/swift_ouster.mcap", "/sensor_station/lidar/points", is_repetitive=True)
#save_point_cloud_ply(src, "robusticpstuff/src.ply")
#save_point_cloud_ply(tgt, "robusticpstuff/tgt.ply")
result = subprocess.run(['./robusticpstuff/FRICP', './robusticpstuff/src.ply', './robusticpstuff/tgt.ply', '/robusticpstuff', '3'],
                       capture_output=True, text=True)
print(result)