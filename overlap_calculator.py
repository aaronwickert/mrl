import numpy as np
from typing import Tuple
from kissmatcher import bag2np


def compute_point_cloud_overlap(
        pc1: np.ndarray,
        pc2: np.ndarray,
        T1: np.ndarray,
        T2: np.ndarray,
        voxel_resolution: float = 0.01
) -> float:
    """
    Compute the spatial overlap ratio between two point clouds.

    Parameters
    ----------
    pc1 : np.ndarray
        First point cloud of shape (N, 3)
    pc2 : np.ndarray
        Second point cloud of shape (M, 3)
    T1 : np.ndarray
        Transformation matrix for pc1 of shape (4, 4)
    T2 : np.ndarray
        Transformation matrix for pc2 of shape (4, 4)
    voxel_resolution : float, optional
        Voxel grid resolution in meters (default: 0.01)

    Returns
    -------
    rho : float
        Overlap ratio in range [0, 1]
    """

    def transform_point_cloud(pc: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply rigid transformation to point cloud."""
        # Convert to homogeneous coordinates
        pc_homogeneous = np.hstack([pc, np.ones((pc.shape[0], 1))])
        # Apply transformation
        pc_transformed = (T @ pc_homogeneous.T).T
        # Return only xyz coordinates
        return pc_transformed[:, :3]

    def voxelize(pc: np.ndarray, resolution: float) -> set:
        """Discretize point cloud into voxel grid."""
        # Compute voxel indices
        voxel_indices = np.floor(pc / resolution).astype(np.int32)
        # Convert to set of tuples for efficient intersection
        voxels = set(map(tuple, voxel_indices))
        return voxels

    # Transform both point clouds to common reference frame
    pc1_transformed = transform_point_cloud(pc1, T1)
    pc2_transformed = transform_point_cloud(pc2, T2)

    # Voxelize both point clouds
    V1 = voxelize(pc1_transformed, voxel_resolution)
    V2 = voxelize(pc2_transformed, voxel_resolution)

    # Compute overlap ratio
    intersection = len(V1 & V2)
    min_cardinality = min(len(V1), len(V2))

    # Handle edge case of empty point clouds
    if min_cardinality == 0:
        return 0.0

    rho = intersection / min_cardinality

    return rho


# Example usage:
if __name__ == "__main__":
    # Generate sample point clouds
    pc1 = np.random.rand(1000, 3) * 10  # 1000 points in 10m cube
    pc2 = np.random.rand(1500, 3) * 10  # 1500 points in 10m cube

    # Identity transformations (both in same frame)
    T1 = np.eye(4)
    T2 = np.eye(4)

    # Compute overlap
    overlap = compute_point_cloud_overlap(pc1, pc2, T1, T2)
    print(f"Overlap ratio: {overlap:.4f}")