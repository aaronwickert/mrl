"""
MCAP Bag Decompression and Data Extraction Script

This script reads compressed ROS2 MCAP bag files, decompresses them,
extracts data from various topics, and creates a comprehensive metadata file.

Input: /home/aaron/uni/thesis/data/seq4/seq4_telemax/
Output: /home/aaron/uni/thesis/data/seq4/seq4_telemax_decompressed/
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
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
    print("Warning: PIL not available, images will be saved as raw numpy arrays")

# For point clouds
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("Warning: Open3D not available, point clouds will be saved as numpy arrays only")


# =============================================================================
# Configuration
# =============================================================================

# Use the decompressed files that already exist
INPUT_DIR = Path("/home/aaron/uni/thesis/data/seq4/seq4_telemax_decompressed")
OUTPUT_DIR = Path("/home/aaron/uni/thesis/data/seq4/seq4_telemax_decompressed")

# Topics to extract
TOPICS_CONFIG = {
    # Point cloud data
    "/telemax/lidar/points": {
        "type": "pointcloud",
        "save_format": ["npy", "pcd"],  # numpy and PCD formats
        "subsample": 1,  # Save every Nth message
    },
    # IMU data
    "/telemax/lidar/imu": {
        "type": "imu",
        "save_format": ["npy", "csv"],
        "subsample": 10,  # IMU is high frequency, subsample
    },
    # Camera images
    "/telemax/arm_rgbd_cam/color/image_raw": {
        "type": "image",
        "save_format": ["png"],
        "subsample": 5,  # Save every 5th frame
    },
    "/telemax/front_rgbd_cam/color/image_raw": {
        "type": "image",
        "save_format": ["png"],
        "subsample": 5,
    },
    # Camera info
    "/telemax/arm_rgbd_cam/color/camera_info": {
        "type": "camera_info",
        "save_format": ["yaml"],
        "subsample": None,  # Only save first message
    },
    "/telemax/front_rgbd_cam/color/camera_info": {
        "type": "camera_info",
        "save_format": ["yaml"],
        "subsample": None,
    },
    # Transforms
    "/telemax/tf": {
        "type": "tf",
        "save_format": ["json"],
        "subsample": 10,
    },
    "/telemax/tf_static": {
        "type": "tf_static",
        "save_format": ["json"],
        "subsample": None,  # Only save once
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TopicStats:
    """Statistics for a single topic."""
    topic_name: str
    message_type: str
    message_count: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    extracted_count: int
    file_paths: List[str]


@dataclass
class BagMetadata:
    """Metadata for the entire bag sequence."""
    source_dir: str
    output_dir: str
    total_bags: int
    total_messages: int
    duration_seconds: float
    start_time_ns: int
    end_time_ns: int
    topics: Dict[str, TopicStats]
    extraction_timestamp: str
    ros_distro: str


# =============================================================================
# Data Extraction Functions
# =============================================================================

def parse_pointcloud2(msg) -> Optional[np.ndarray]:
    """Parse PointCloud2 message to numpy array."""
    try:
        # Get field info
        fields = {f.name: f for f in msg.fields}

        point_step = msg.point_step
        row_step = msg.row_step
        data = bytes(msg.data)

        n_points = msg.width * msg.height

        # Check for common field layouts
        if 'x' in fields and 'y' in fields and 'z' in fields:
            x_offset = fields['x'].offset
            y_offset = fields['y'].offset
            z_offset = fields['z'].offset

            points = np.zeros((n_points, 3), dtype=np.float32)

            for i in range(n_points):
                base = i * point_step
                points[i, 0] = struct.unpack('f', data[base + x_offset:base + x_offset + 4])[0]
                points[i, 1] = struct.unpack('f', data[base + y_offset:base + y_offset + 4])[0]
                points[i, 2] = struct.unpack('f', data[base + z_offset:base + z_offset + 4])[0]

            # Filter out invalid points (NaN, Inf, or very far)
            valid_mask = np.isfinite(points).all(axis=1) & (np.abs(points).max(axis=1) < 1000)
            points = points[valid_mask]

            return points
        else:
            print(f"    Warning: Unknown point cloud field layout: {list(fields.keys())}")
            return None

    except Exception as e:
        print(f"    Error parsing point cloud: {e}")
        return None


def parse_image(msg) -> Optional[np.ndarray]:
    """Parse Image message to numpy array."""
    try:
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        data = bytes(msg.data)

        if encoding in ['rgb8', 'bgr8']:
            img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if encoding == 'bgr8':
                img = img[:, :, ::-1]  # BGR to RGB
        elif encoding == 'mono8':
            img = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        elif encoding == '16UC1':
            img = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        elif encoding == '32FC1':
            img = np.frombuffer(data, dtype=np.float32).reshape(height, width)
        else:
            print(f"    Warning: Unknown image encoding: {encoding}")
            return None

        return img

    except Exception as e:
        print(f"    Error parsing image: {e}")
        return None


def parse_imu(msg) -> Dict[str, Any]:
    """Parse IMU message to dictionary."""
    return {
        'timestamp_sec': msg.header.stamp.sec,
        'timestamp_nanosec': msg.header.stamp.nanosec,
        'orientation': {
            'x': msg.orientation.x,
            'y': msg.orientation.y,
            'z': msg.orientation.z,
            'w': msg.orientation.w,
        },
        'angular_velocity': {
            'x': msg.angular_velocity.x,
            'y': msg.angular_velocity.y,
            'z': msg.angular_velocity.z,
        },
        'linear_acceleration': {
            'x': msg.linear_acceleration.x,
            'y': msg.linear_acceleration.y,
            'z': msg.linear_acceleration.z,
        },
    }


def parse_camera_info(msg) -> Dict[str, Any]:
    """Parse CameraInfo message to dictionary."""
    return {
        'width': msg.width,
        'height': msg.height,
        'distortion_model': msg.distortion_model,
        'D': list(msg.d),
        'K': list(msg.k),
        'R': list(msg.r),
        'P': list(msg.p),
        'binning_x': msg.binning_x,
        'binning_y': msg.binning_y,
        'roi': {
            'x_offset': msg.roi.x_offset,
            'y_offset': msg.roi.y_offset,
            'height': msg.roi.height,
            'width': msg.roi.width,
            'do_rectify': msg.roi.do_rectify,
        }
    }


def parse_tf(msg) -> List[Dict[str, Any]]:
    """Parse TFMessage to list of transforms."""
    transforms = []
    for t in msg.transforms:
        transforms.append({
            'timestamp_sec': t.header.stamp.sec,
            'timestamp_nanosec': t.header.stamp.nanosec,
            'frame_id': t.header.frame_id,
            'child_frame_id': t.child_frame_id,
            'translation': {
                'x': t.transform.translation.x,
                'y': t.transform.translation.y,
                'z': t.transform.translation.z,
            },
            'rotation': {
                'x': t.transform.rotation.x,
                'y': t.transform.rotation.y,
                'z': t.transform.rotation.z,
                'w': t.transform.rotation.w,
            },
        })
    return transforms


# =============================================================================
# Save Functions
# =============================================================================

def save_pointcloud(points: np.ndarray, base_path: Path, formats: List[str]):
    """Save point cloud in multiple formats."""
    paths = []

    if 'npy' in formats:
        npy_path = base_path.with_suffix('.npy')
        np.save(npy_path, points)
        paths.append(str(npy_path))

    if 'pcd' in formats and HAS_O3D:
        pcd_path = base_path.with_suffix('.pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        paths.append(str(pcd_path))

    return paths


def save_image(img: np.ndarray, base_path: Path, formats: List[str]):
    """Save image in multiple formats."""
    paths = []

    if 'png' in formats and HAS_PIL:
        png_path = base_path.with_suffix('.png')
        if len(img.shape) == 2:
            Image.fromarray(img).save(png_path)
        else:
            Image.fromarray(img).save(png_path)
        paths.append(str(png_path))
    elif 'npy' in formats or not HAS_PIL:
        npy_path = base_path.with_suffix('.npy')
        np.save(npy_path, img)
        paths.append(str(npy_path))

    return paths


# =============================================================================
# Main Processing
# =============================================================================

def process_bags(input_dir: Path, output_dir: Path) -> BagMetadata:
    """Process all MCAP bags in the input directory."""

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {
        'pointclouds': output_dir / 'pointclouds',
        'images': output_dir / 'images',
        'imu': output_dir / 'imu',
        'camera_info': output_dir / 'camera_info',
        'transforms': output_dir / 'transforms',
    }
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)

    # Find all MCAP files
    mcap_files = sorted(input_dir.glob("*.mcap"))
    print(f"Found {len(mcap_files)} MCAP files")

    # Initialize counters and storage
    topic_stats: Dict[str, Dict] = defaultdict(lambda: {
        'message_count': 0,
        'extracted_count': 0,
        'first_timestamp_ns': None,
        'last_timestamp_ns': None,
        'message_type': '',
        'file_paths': [],
    })

    topic_counters: Dict[str, int] = defaultdict(int)
    total_messages = 0
    global_first_ts = None
    global_last_ts = None

    # Accumulated data for batch saving
    imu_data: Dict[str, List] = defaultdict(list)
    tf_data: List[Dict] = []
    tf_static_data: List[Dict] = []
    camera_info_saved: Dict[str, bool] = {}

    # Process each MCAP file
    for mcap_idx, mcap_path in enumerate(mcap_files):
        print(f"\n[{mcap_idx + 1}/{len(mcap_files)}] Processing: {mcap_path.name}")

        try:
            with open(mcap_path, 'rb') as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])

                msg_count_in_file = 0
                error_count_in_file = 0

                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    msg_count_in_file += 1
                    topic = channel.topic
                    timestamp_ns = message.log_time
                    total_messages += 1

                    # Update global timestamps
                    if global_first_ts is None or timestamp_ns < global_first_ts:
                        global_first_ts = timestamp_ns
                    if global_last_ts is None or timestamp_ns > global_last_ts:
                        global_last_ts = timestamp_ns

                    # Update topic stats
                    stats = topic_stats[topic]
                    stats['message_count'] += 1
                    stats['message_type'] = channel.message_encoding
                    if stats['first_timestamp_ns'] is None:
                        stats['first_timestamp_ns'] = timestamp_ns
                    stats['last_timestamp_ns'] = timestamp_ns

                    # Check if we should process this topic
                    if topic not in TOPICS_CONFIG:
                        continue

                    config = TOPICS_CONFIG[topic]
                    topic_counters[topic] += 1
                    counter = topic_counters[topic]

                    # Check subsample
                    subsample = config.get('subsample', 1)
                    if subsample is None:
                        # Only process first message
                        if stats['extracted_count'] > 0:
                            continue
                    elif counter % subsample != 0:
                        continue

                    # Process based on type
                    data_type = config['type']
                    formats = config['save_format']

                    try:
                        if data_type == 'pointcloud':
                            points = parse_pointcloud2(ros_msg)
                            if points is not None and len(points) > 0:
                                base_name = f"pointcloud_{stats['extracted_count']:06d}"
                                base_path = subdirs['pointclouds'] / base_name
                                paths = save_pointcloud(points, base_path, formats)
                                stats['file_paths'].extend(paths)
                                stats['extracted_count'] += 1

                        elif data_type == 'image':
                            img = parse_image(ros_msg)
                            if img is not None:
                                # Create subfolder for each camera
                                cam_name = topic.split('/')[-3]  # e.g., arm_rgbd_cam
                                cam_dir = subdirs['images'] / cam_name
                                cam_dir.mkdir(exist_ok=True)
                                base_name = f"image_{stats['extracted_count']:06d}"
                                base_path = cam_dir / base_name
                                paths = save_image(img, base_path, formats)
                                stats['file_paths'].extend(paths)
                                stats['extracted_count'] += 1

                        elif data_type == 'imu':
                            imu_dict = parse_imu(ros_msg)
                            imu_data[topic].append(imu_dict)
                            stats['extracted_count'] += 1

                        elif data_type == 'camera_info':
                            if topic not in camera_info_saved:
                                cam_info = parse_camera_info(ros_msg)
                                cam_name = topic.split('/')[-3]
                                cam_info_path = subdirs['camera_info'] / f"{cam_name}_camera_info.yaml"
                                with open(cam_info_path, 'w') as f:
                                    yaml.dump(cam_info, f, default_flow_style=False)
                                stats['file_paths'].append(str(cam_info_path))
                                stats['extracted_count'] += 1
                                camera_info_saved[topic] = True

                        elif data_type == 'tf':
                            transforms = parse_tf(ros_msg)
                            tf_data.extend(transforms)
                            stats['extracted_count'] += 1

                        elif data_type == 'tf_static':
                            if stats['extracted_count'] == 0:
                                transforms = parse_tf(ros_msg)
                                tf_static_data.extend(transforms)
                                stats['extracted_count'] += 1

                    except Exception as e:
                        print(f"    Error processing {topic}: {e}")
                        continue

                print(f"  Processed {msg_count_in_file} messages, {error_count_in_file} errors")

        except Exception as e:
            print(f"  Error reading {mcap_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Progress update
        print(f"  Total so far: {total_messages} messages")

    # Save accumulated data
    print("\nSaving accumulated data...")

    # Save IMU data
    for topic, data in imu_data.items():
        if data:
            imu_path = subdirs['imu'] / f"imu_data.json"
            with open(imu_path, 'w') as f:
                json.dump(data, f, indent=2)
            topic_stats[topic]['file_paths'].append(str(imu_path))

            # Also save as numpy arrays for easier processing
            timestamps = np.array([[d['timestamp_sec'], d['timestamp_nanosec']] for d in data])
            angular_vel = np.array([[d['angular_velocity']['x'],
                                     d['angular_velocity']['y'],
                                     d['angular_velocity']['z']] for d in data])
            linear_acc = np.array([[d['linear_acceleration']['x'],
                                    d['linear_acceleration']['y'],
                                    d['linear_acceleration']['z']] for d in data])

            np.save(subdirs['imu'] / 'imu_timestamps.npy', timestamps)
            np.save(subdirs['imu'] / 'imu_angular_velocity.npy', angular_vel)
            np.save(subdirs['imu'] / 'imu_linear_acceleration.npy', linear_acc)

    # Save TF data
    if tf_data:
        tf_path = subdirs['transforms'] / 'tf_data.json'
        with open(tf_path, 'w') as f:
            json.dump(tf_data, f, indent=2)
        topic_stats['/telemax/tf']['file_paths'].append(str(tf_path))

    if tf_static_data:
        tf_static_path = subdirs['transforms'] / 'tf_static_data.json'
        with open(tf_static_path, 'w') as f:
            json.dump(tf_static_data, f, indent=2)
        topic_stats['/telemax/tf_static']['file_paths'].append(str(tf_static_path))

    # Create metadata
    duration_ns = global_last_ts - global_first_ts if global_first_ts and global_last_ts else 0

    metadata = BagMetadata(
        source_dir=str(input_dir),
        output_dir=str(output_dir),
        total_bags=len(mcap_files),
        total_messages=total_messages,
        duration_seconds=duration_ns / 1e9,
        start_time_ns=global_first_ts or 0,
        end_time_ns=global_last_ts or 0,
        topics={
            topic: TopicStats(
                topic_name=topic,
                message_type=stats['message_type'],
                message_count=stats['message_count'],
                first_timestamp_ns=stats['first_timestamp_ns'] or 0,
                last_timestamp_ns=stats['last_timestamp_ns'] or 0,
                extracted_count=stats['extracted_count'],
                file_paths=stats['file_paths'],
            )
            for topic, stats in topic_stats.items()
        },
        extraction_timestamp=datetime.now().isoformat(),
        ros_distro='jazzy',
    )

    return metadata


def save_metadata(metadata: BagMetadata, output_dir: Path):
    """Save metadata to JSON and YAML files."""

    # Convert to dict
    metadata_dict = {
        'source_dir': metadata.source_dir,
        'output_dir': metadata.output_dir,
        'total_bags': metadata.total_bags,
        'total_messages': metadata.total_messages,
        'duration_seconds': metadata.duration_seconds,
        'start_time_ns': metadata.start_time_ns,
        'end_time_ns': metadata.end_time_ns,
        'extraction_timestamp': metadata.extraction_timestamp,
        'ros_distro': metadata.ros_distro,
        'topics': {
            topic: asdict(stats)
            for topic, stats in metadata.topics.items()
        }
    }

    # Save as JSON
    json_path = output_dir / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"Saved metadata to {json_path}")

    # Save as YAML
    yaml_path = output_dir / 'metadata.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata_dict, f, default_flow_style=False)
    print(f"Saved metadata to {yaml_path}")

    # Save summary
    summary_path = output_dir / 'README.md'
    with open(summary_path, 'w') as f:
        f.write("# Decompressed MCAP Bag Data\n\n")
        f.write(f"**Source:** `{metadata.source_dir}`\n\n")
        f.write(f"**Extraction Date:** {metadata.extraction_timestamp}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Bags:** {metadata.total_bags}\n")
        f.write(f"- **Total Messages:** {metadata.total_messages:,}\n")
        f.write(f"- **Duration:** {metadata.duration_seconds:.2f} seconds ({metadata.duration_seconds/60:.2f} minutes)\n")
        f.write(f"- **ROS Distro:** {metadata.ros_distro}\n\n")

        f.write("## Topics\n\n")
        f.write("| Topic | Type | Messages | Extracted |\n")
        f.write("|-------|------|----------|----------|\n")
        for topic, stats in sorted(metadata.topics.items()):
            f.write(f"| `{topic}` | {stats.message_type} | {stats.message_count:,} | {stats.extracted_count:,} |\n")

        f.write("\n## Directory Structure\n\n")
        f.write("```\n")
        f.write("seq4_telemax_decompressed/\n")
        f.write("├── pointclouds/          # LiDAR point clouds (.npy, .pcd)\n")
        f.write("├── images/               # Camera images (.png)\n")
        f.write("│   ├── arm_rgbd_cam/\n")
        f.write("│   └── front_rgbd_cam/\n")
        f.write("├── imu/                  # IMU data (.json, .npy)\n")
        f.write("├── camera_info/          # Camera calibration (.yaml)\n")
        f.write("├── transforms/           # TF data (.json)\n")
        f.write("├── metadata.json\n")
        f.write("├── metadata.yaml\n")
        f.write("└── README.md\n")
        f.write("```\n")

    print(f"Saved summary to {summary_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MCAP Bag Decompression and Data Extraction")
    print("=" * 60)
    print(f"\nInput:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Check input exists
    if not INPUT_DIR.exists():
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    # Process bags
    metadata = process_bags(INPUT_DIR, OUTPUT_DIR)

    # Save metadata
    print("\n" + "=" * 60)
    print("Saving metadata...")
    save_metadata(metadata, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal messages processed: {metadata.total_messages:,}")
    print(f"Duration: {metadata.duration_seconds:.2f} seconds")
    print(f"\nExtracted data by topic:")
    for topic, stats in sorted(metadata.topics.items()):
        if stats.extracted_count > 0:
            print(f"  {topic}: {stats.extracted_count:,} items")

    print(f"\nOutput saved to: {OUTPUT_DIR}")
