std_msgs/Header header

# Keypoints as float32 xyz coordinates (metres)
# keypoint_count = keypoint_xyz.size() / 3
float32[] keypoint_xyz

# Descriptors as float32 values
# Size: (keypoint_xyz.size() / 3) * descriptor_dim
float32[] descriptors
uint32 descriptor_dim
