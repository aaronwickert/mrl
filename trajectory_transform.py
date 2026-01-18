import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def compute_trajectory_transformation(
    trajectory_source: np.ndarray,
    trajectory_target: np.ndarray,
    allow_scaling: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the rigid transformation (rotation, translation) between two trajectories
    using the Kabsch algorithm (SVD-based).

    Parameters
    ----------
    trajectory_source : np.ndarray
        Source trajectory, shape (N, M) where M >= 3. First 3 columns are x, y, z.
    trajectory_target : np.ndarray
        Target trajectory, shape (N, M) where M >= 3. First 3 columns are x, y, z.
    allow_scaling : bool
        If True, also compute optimal uniform scaling factor.

    Returns
    -------
    R : np.ndarray
        3x3 rotation matrix
    t : np.ndarray
        3x1 translation vector
    scale : float
        Scaling factor (1.0 if allow_scaling is False)
    rmse : float
        Root mean square error after transformation

    The transformation applies as: target ≈ scale * (source @ R.T) + t
    """
    # Extract xyz coordinates
    P = trajectory_source[:, :3].astype(np.float64)
    Q = trajectory_target[:, :3].astype(np.float64)

    if P.shape[0] != Q.shape[0]:
        raise ValueError(f"Trajectories must have same number of points. Got {P.shape[0]} and {Q.shape[0]}")

    if P.shape[0] < 3:
        raise ValueError("Need at least 3 points to compute transformation")

    # Step 1: Compute centroids
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    # Step 2: Center the point sets
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Step 3: Compute covariance matrix
    H = P_centered.T @ Q_centered

    # Step 4: SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Compute rotation matrix
    R = Vt.T @ U.T

    # Step 6: Handle reflection case (ensure proper rotation, det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 7: Compute scaling if requested
    scale = 1.0
    if allow_scaling:
        # Optimal scale: sum(S) / sum(P_centered^2)
        scale = S.sum() / (P_centered ** 2).sum()

    # Step 8: Compute translation
    t = centroid_Q - scale * (R @ centroid_P)

    # Step 9: Compute RMSE
    P_transformed = scale * (P @ R.T) + t
    rmse = np.sqrt(((P_transformed - Q) ** 2).sum(axis=1).mean())

    return R, t, scale, rmse


def apply_transformation(
    trajectory: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """Apply transformation to trajectory. Only transforms xyz, preserves other columns."""
    result = trajectory.copy()
    result[:, :3] = scale * (trajectory[:, :3] @ R.T) + t
    return result


TRAJECTORY_COLUMN = "/world/trajectory:LineStrips3D:strips"


def extract_trajectory_from_dataframe(
    df: pd.DataFrame,
    column: str = TRAJECTORY_COLUMN,
    strip_index: int = 0,
    concatenate_strips: bool = True
) -> np.ndarray:
    """
    Extract trajectory data from a pandas DataFrame containing Rerun LineStrips3D data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Rerun trajectory data.
    column : str
        Column name containing LineStrips3D strips data.
        Default: "/world/trajectory:LineStrips3D:strips"
    strip_index : int
        If concatenate_strips is False, which strip index to extract.
    concatenate_strips : bool
        If True, concatenate all strips into a single trajectory.
        If False, extract only the strip at strip_index.

    Returns
    -------
    np.ndarray
        Trajectory as (N, 3) array of xyz coordinates.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    points_list: List[np.ndarray] = []

    for row_data in df[column].dropna():
        strips = _extract_strips(row_data)

        if concatenate_strips:
            for strip in strips:
                points_list.append(strip)
        else:
            if strip_index < len(strips):
                points_list.append(strips[strip_index])

    if not points_list:
        raise ValueError(f"No valid trajectory data found in column '{column}'")

    return np.vstack(points_list)


def _extract_strips(data) -> List[np.ndarray]:
    """Extract strips from various Rerun data formats."""
    # Handle pyarrow ListArray (common when reading from Rerun dataframe)
    if hasattr(data, 'as_py'):
        data = data.as_py()

    # Handle None/empty
    if data is None:
        return []

    # Handle numpy array
    if isinstance(data, np.ndarray):
        # Case: 2D array with xyz columns (N, 3+)
        if data.ndim == 2 and data.shape[1] >= 3:
            return [data[:, :3].astype(np.float64)]
        # Case: 3D array of strips (S, N, 3)
        elif data.ndim == 3:
            return [strip[:, :3].astype(np.float64) for strip in data]
        # Case: object array (Rerun LineStrips3D format)
        elif data.dtype == object:
            strips = []
            for item in data:
                if item is None:
                    continue
                # Item is an array of point arrays
                if isinstance(item, np.ndarray) and item.dtype == object:
                    points = np.array([np.asarray(p, dtype=np.float64) for p in item])
                    if points.ndim == 2 and points.shape[1] >= 3:
                        strips.append(points[:, :3])
                elif isinstance(item, np.ndarray) and item.ndim == 2:
                    strips.append(item[:, :3].astype(np.float64))
            return strips
        # Case: 1D array (single point)
        elif data.ndim == 1 and len(data) >= 3:
            return [data[:3].reshape(1, 3).astype(np.float64)]

    # List of strips (list of lists/arrays)
    if isinstance(data, (list, tuple)):
        strips = []
        for item in data:
            if item is None:
                continue
            # Handle pyarrow nested structures
            if hasattr(item, 'as_py'):
                item = item.as_py()
            if hasattr(item, 'values'):
                item = item.values

            arr = np.asarray(item, dtype=np.float64)
            if arr.ndim == 1 and len(arr) >= 3:
                # Single point as flat array
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                strips.append(arr[:, :3])
        return strips

    raise ValueError(f"Unsupported data format: {type(data)}")


if __name__ == "__main__":
    # Example usage
    source = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    target = np.array([[1, 1, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]], dtype=float)

    R, t, scale, rmse = compute_trajectory_transformation(source, target)
    print(f"Rotation matrix:\n{R}")
    print(f"Translation: {t}")
    print(f"RMSE: {rmse:.6f}")

    aligned = apply_transformation(source, R, t, scale)
    print(f"Aligned trajectory:\n{aligned}")
