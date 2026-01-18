"""
MCAP Data Extraction Script - Robust Version

This script reads ROS2 MCAP bag files and extracts point cloud, IMU, and image data
with robust error handling for corrupted messages.

Uses the raw mcap library to read messages and manually deserializes CDR data
for better control over error handling.
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import struct

# MCAP reading
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# For images
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# For point clouds
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path("/home/aaron/uni/thesis/data/seq4/seq4_telemax")
OUTPUT_DIR = Path("/home/aaron/uni/thesis/data/seq4/seq4_telemax_decompressed")


# =============================================================================
# CDR Deserialization Functions (Manual Implementation)
# =============================================================================

def read_uint32(data: bytes, offset: int) -> Tuple[int, int]:
    """Read uint32 from CDR data."""
    # Align to 4 bytes
    offset = (offset + 3) & ~3
    value = struct.unpack_from('<I', data, offset)[0]
    return value, offset + 4


def read_int32(data: bytes, offset: int) -> Tuple[int, int]:
    """Read int32 from CDR data."""
    offset = (offset + 3) & ~3
    value = struct.unpack_from('<i', data, offset)[0]
    return value, offset + 4


def read_float32(data: bytes, offset: int) -> Tuple[float, int]:
    """Read float32 from CDR data."""
    offset = (offset + 3) & ~3
    value = struct.unpack_from('<f', data, offset)[0]
    return value, offset + 4


def read_float64(data: bytes, offset: int) -> Tuple[float, int]:
    """Read float64 from CDR data."""
    offset = (offset + 7) & ~7
    value = struct.unpack_from('<d', data, offset)[0]
    return value, offset + 8


def read_uint8(data: bytes, offset: int) -> Tuple[int, int]:
    """Read uint8 from CDR data."""
    return data[offset], offset + 1


def read_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Read string from CDR data with error handling."""
    length, offset = read_uint32(data, offset)
    if length == 0:
        return "", offset
    try:
        # Try UTF-8 first
        value = data[offset:offset + length - 1].decode('utf-8')
    except UnicodeDecodeError:
        # Fall back to latin-1 or just return placeholder
        try:
            value = data[offset:offset + length - 1].decode('latin-1')
        except:
            value = f"<binary:{length}bytes>"
    return value, offset + length


def read_header(data: bytes, offset: int) -> Tuple[Dict, int]:
    """Read std_msgs/Header from CDR data."""
    # Skip CDR encapsulation header (4 bytes)
    if offset == 0:
        offset = 4

    header = {}

    # stamp (builtin_interfaces/Time)
    sec, offset = read_int32(data, offset)
    nanosec, offset = read_uint32(data, offset)
    header['stamp'] = {'sec': sec, 'nanosec': nanosec}

    # frame_id (string)
    frame_id, offset = read_string(data, offset)
    header['frame_id'] = frame_id

    return header, offset


def parse_pointcloud2_raw(data: bytes) -> Optional[np.ndarray]:
    """Parse PointCloud2 message from raw CDR data."""
    try:
        offset = 4  # Skip CDR header

        # Header
        header, offset = read_header(data, offset)

        # height, width
        height, offset = read_uint32(data, offset)
        width, offset = read_uint32(data, offset)

        # fields (array)
        num_fields, offset = read_uint32(data, offset)
        fields = {}
        for _ in range(num_fields):
            name, offset = read_string(data, offset)
            field_offset, offset = read_uint32(data, offset)
            datatype, offset = read_uint8(data, offset)
            count, offset = read_uint32(data, offset)
            fields[name] = {'offset': field_offset, 'datatype': datatype, 'count': count}

        # is_bigendian
        is_bigendian, offset = read_uint8(data, offset)

        # point_step
        point_step, offset = read_uint32(data, offset)

        # row_step
        row_step, offset = read_uint32(data, offset)

        # data (byte array)
        data_length, offset = read_uint32(data, offset)
        point_data = data[offset:offset + data_length]

        # Extract XYZ points
        n_points = width * height

        if 'x' in fields and 'y' in fields and 'z' in fields:
            x_off = fields['x']['offset']
            y_off = fields['y']['offset']
            z_off = fields['z']['offset']

            points = np.zeros((n_points, 3), dtype=np.float32)

            for i in range(n_points):
                base = i * point_step
                if base + z_off + 4 <= len(point_data):
                    points[i, 0] = struct.unpack_from('<f', point_data, base + x_off)[0]
                    points[i, 1] = struct.unpack_from('<f', point_data, base + y_off)[0]
                    points[i, 2] = struct.unpack_from('<f', point_data, base + z_off)[0]

            # Filter invalid points
            valid_mask = np.isfinite(points).all(axis=1) & (np.abs(points).max(axis=1) < 1000)
            points = points[valid_mask]

            return points

        return None

    except Exception as e:
        return None


def parse_imu_raw(data: bytes) -> Optional[Dict]:
    """Parse IMU message from raw CDR data."""
    try:
        offset = 4  # Skip CDR header

        # Header
        header, offset = read_header(data, offset)

        # orientation (Quaternion)
        ox, offset = read_float64(data, offset)
        oy, offset = read_float64(data, offset)
        oz, offset = read_float64(data, offset)
        ow, offset = read_float64(data, offset)

        # orientation_covariance (9 doubles)
        offset += 72

        # angular_velocity (Vector3)
        avx, offset = read_float64(data, offset)
        avy, offset = read_float64(data, offset)
        avz, offset = read_float64(data, offset)

        # angular_velocity_covariance (9 doubles)
        offset += 72

        # linear_acceleration (Vector3)
        lax, offset = read_float64(data, offset)
        lay, offset = read_float64(data, offset)
        laz, offset = read_float64(data, offset)

        return {
            'timestamp_sec': header['stamp']['sec'],
            'timestamp_nanosec': header['stamp']['nanosec'],
            'orientation': {'x': ox, 'y': oy, 'z': oz, 'w': ow},
            'angular_velocity': {'x': avx, 'y': avy, 'z': avz},
            'linear_acceleration': {'x': lax, 'y': lay, 'z': laz},
        }

    except Exception as e:
        return None


def parse_image_raw(data: bytes) -> Optional[Tuple[np.ndarray, Dict]]:
    """Parse Image message from raw CDR data."""
    try:
        offset = 4  # Skip CDR header

        # Header
        header, offset = read_header(data, offset)

        # height, width
        height, offset = read_uint32(data, offset)
        width, offset = read_uint32(data, offset)

        # encoding
        encoding, offset = read_string(data, offset)

        # is_bigendian
        is_bigendian, offset = read_uint8(data, offset)

        # step
        step, offset = read_uint32(data, offset)

        # data
        data_length, offset = read_uint32(data, offset)
        img_data = data[offset:offset + data_length]

        # Convert to numpy array
        if encoding in ['rgb8', 'bgr8']:
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
            if encoding == 'bgr8':
                img = img[:, :, ::-1]
        elif encoding == 'mono8':
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width)
        else:
            return None

        return img, {'encoding': encoding, 'width': width, 'height': height,
                     'timestamp': header['stamp']}

    except Exception as e:
        return None


# =============================================================================
# Data Storage Classes
# =============================================================================

@dataclass
class ExtractionStats:
    topic_name: str
    message_type: str
    total_count: int = 0
    extracted_count: int = 0
    error_count: int = 0
    file_paths: List[str] = field(default_factory=list)


# =============================================================================
# Main Processing
# =============================================================================

def process_bags(input_dir: Path, output_dir: Path):
    """Process all MCAP bags and extract data."""

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {
        'pointclouds': output_dir / 'pointclouds',
        'imu': output_dir / 'imu',
        'images': output_dir / 'images',
    }
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)

    # Find MCAP files
    mcap_files = sorted(input_dir.glob("*.mcap"))
    print(f"Found {len(mcap_files)} MCAP files")

    # Storage for accumulated data
    stats: Dict[str, ExtractionStats] = defaultdict(lambda: ExtractionStats("", ""))
    imu_data: List[Dict] = []
    pointcloud_count = 0
    image_counts: Dict[str, int] = defaultdict(int)

    total_messages = 0

    # Process each file
    for mcap_idx, mcap_path in enumerate(mcap_files):
        print(f"\n[{mcap_idx + 1}/{len(mcap_files)}] Processing: {mcap_path.name}")

        try:
            with open(mcap_path, 'rb') as f:
                reader = make_reader(f)

                # Read raw messages without decoding
                for schema, channel, message in reader.iter_messages():
                    topic = channel.topic
                    msg_data = message.data
                    total_messages += 1

                    # Initialize stats
                    if stats[topic].topic_name == "":
                        stats[topic].topic_name = topic
                        stats[topic].message_type = schema.name if schema else "unknown"

                    stats[topic].total_count += 1

                    try:
                        # Process based on topic
                        if topic == '/telemax/lidar/points':
                            points = parse_pointcloud2_raw(msg_data)
                            if points is not None and len(points) > 100:
                                # Save point cloud
                                pc_path = subdirs['pointclouds'] / f'pointcloud_{pointcloud_count:06d}.npy'
                                np.save(pc_path, points)
                                stats[topic].file_paths.append(str(pc_path))
                                stats[topic].extracted_count += 1
                                pointcloud_count += 1

                                if pointcloud_count % 100 == 0:
                                    print(f"  Saved {pointcloud_count} point clouds...")

                        elif topic == '/telemax/lidar/imu':
                            # Subsample IMU (every 10th message)
                            if stats[topic].total_count % 10 == 0:
                                imu = parse_imu_raw(msg_data)
                                if imu is not None:
                                    imu_data.append(imu)
                                    stats[topic].extracted_count += 1

                        elif 'image_raw' in topic:
                            # Subsample images (every 5th)
                            if stats[topic].total_count % 5 == 0:
                                result = parse_image_raw(msg_data)
                                if result is not None:
                                    img, info = result
                                    cam_name = topic.split('/')[-3]
                                    cam_dir = subdirs['images'] / cam_name
                                    cam_dir.mkdir(exist_ok=True)

                                    img_path = cam_dir / f'image_{image_counts[cam_name]:06d}.png'
                                    if HAS_PIL:
                                        Image.fromarray(img).save(img_path)
                                    else:
                                        np.save(img_path.with_suffix('.npy'), img)

                                    stats[topic].file_paths.append(str(img_path))
                                    stats[topic].extracted_count += 1
                                    image_counts[cam_name] += 1

                    except Exception as e:
                        stats[topic].error_count += 1
                        continue

                print(f"  Processed messages, total so far: {total_messages}")

        except Exception as e:
            print(f"  Error reading file: {e}")
            continue

    # Save IMU data
    print(f"\nSaving IMU data ({len(imu_data)} samples)...")
    if imu_data:
        imu_json_path = subdirs['imu'] / 'imu_data.json'
        with open(imu_json_path, 'w') as f:
            json.dump(imu_data, f, indent=2)

        # Also save as numpy arrays
        timestamps = np.array([[d['timestamp_sec'], d['timestamp_nanosec']] for d in imu_data])
        angular_vel = np.array([[d['angular_velocity']['x'],
                                 d['angular_velocity']['y'],
                                 d['angular_velocity']['z']] for d in imu_data])
        linear_acc = np.array([[d['linear_acceleration']['x'],
                                d['linear_acceleration']['y'],
                                d['linear_acceleration']['z']] for d in imu_data])
        orientations = np.array([[d['orientation']['x'], d['orientation']['y'],
                                  d['orientation']['z'], d['orientation']['w']] for d in imu_data])

        np.save(subdirs['imu'] / 'imu_timestamps.npy', timestamps)
        np.save(subdirs['imu'] / 'imu_angular_velocity.npy', angular_vel)
        np.save(subdirs['imu'] / 'imu_linear_acceleration.npy', linear_acc)
        np.save(subdirs['imu'] / 'imu_orientation.npy', orientations)

        stats['/telemax/lidar/imu'].file_paths.append(str(imu_json_path))

    # Create metadata
    print("\nCreating metadata...")

    metadata = {
        'source_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_bags': len(mcap_files),
        'total_messages': total_messages,
        'extraction_timestamp': datetime.now().isoformat(),
        'topics': {}
    }

    for topic, s in stats.items():
        if s.total_count > 0:
            metadata['topics'][topic] = {
                'message_type': s.message_type,
                'total_count': s.total_count,
                'extracted_count': s.extracted_count,
                'error_count': s.error_count,
                'num_files': len(s.file_paths),
            }

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    # Create README
    with open(output_dir / 'README.md', 'w') as f:
        f.write("# Extracted MCAP Bag Data\n\n")
        f.write(f"**Source:** `{input_dir}`\n\n")
        f.write(f"**Extraction Date:** {metadata['extraction_timestamp']}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Bags:** {len(mcap_files)}\n")
        f.write(f"- **Total Messages:** {total_messages:,}\n\n")

        f.write("## Extracted Data\n\n")
        f.write("| Topic | Type | Total | Extracted | Errors |\n")
        f.write("|-------|------|-------|-----------|--------|\n")
        for topic, info in sorted(metadata['topics'].items()):
            f.write(f"| `{topic}` | {info['message_type']} | {info['total_count']:,} | {info['extracted_count']:,} | {info['error_count']:,} |\n")

        f.write("\n## Directory Structure\n\n")
        f.write("```\n")
        f.write(f"{output_dir.name}/\n")
        f.write("├── pointclouds/     # LiDAR point clouds (.npy)\n")
        f.write("├── imu/             # IMU data (.json, .npy)\n")
        f.write("├── images/          # Camera images (.png)\n")
        f.write("│   ├── arm_rgbd_cam/\n")
        f.write("│   └── front_rgbd_cam/\n")
        f.write("├── metadata.json\n")
        f.write("├── metadata.yaml\n")
        f.write("└── README.md\n")
        f.write("```\n")

    print(f"\nDone! Output saved to: {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for topic, s in sorted(stats.items()):
        if s.extracted_count > 0:
            print(f"  {topic}: {s.extracted_count:,} extracted ({s.error_count} errors)")


if __name__ == "__main__":
    print("=" * 60)
    print("MCAP Data Extraction")
    print("=" * 60)

    process_bags(INPUT_DIR, OUTPUT_DIR)
