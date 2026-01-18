"""
Transformation Tree for Point Cloud Registration Evaluation

This module visualizes the coordinate frame relationships used in eval.py
for the Swift robot and Ouster sensor station registration evaluation.

Coordinate Frames:
==================
1. MoCap World Frame (W)     - Global reference from motion capture system
2. Swift MoCap Frame (S_m)   - Swift robot pose in MoCap world
3. Ouster MoCap Frame (O_m)  - Ouster station pose in MoCap world
4. Swift Body Frame (S_b)    - Swift robot body frame
5. Swift Front LiDAR (S_fl)  - Front LiDAR sensor frame
6. Swift Back LiDAR (S_bl)   - Back LiDAR sensor frame
7. Ouster LiDAR Frame (O_l)  - Ouster sensor LiDAR frame

Transformations:
================
- T_W_Sm  : MoCap World -> Swift MoCap (from Qualisys + SWIFT_MOCAP_OFFSET)
- T_W_Om  : MoCap World -> Ouster MoCap (from Qualisys)
- T_Sb_Sfl: Swift Body -> Swift Front LiDAR (SWIFT_FRONT_LIDAR_FRAME)
- T_Sb_Sbl: Swift Body -> Swift Back LiDAR (SWIFT_BACK_LIDAR_FRAME)
- T_gt    : Ground truth Swift -> Ouster (relative transform)
- T_est   : Estimated transform from registration algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import Dict, Tuple, Optional


# =============================================================================
# Transformation Constants (from eval.py)
# =============================================================================

SWIFT_MOCAP_OFFSET = np.array([
    [0.5453810259462297, -0.8020423573026362, 0.24349043847812943, 0],
    [0.7342469812568727, 0.31703428250476934, -0.6003087824044075, 0],
    [0.4042782544894191, 0.5061791390497241, 0.7618016619421807, 0],
    [0.0, 0.0, 0.0, 1.0],
])

SWIFT_FRONT_LIDAR_FRAME = np.array([
    [-np.sqrt(3)/2,  0.0, -0.5,  0.19],
    [ 0.0,          -1.0,  0.0,  0.0],
    [ 0.5,           0.0,  np.sqrt(3)/2, 0.2204],
    [ 0.0,           0.0,  0.0,  1.0]
])

SWIFT_BACK_LIDAR_FRAME = np.array([
    [ np.sqrt(3)/2,  0.0, -0.5, -0.19],
    [ 0.0,           1.0,  0.0,  0.0],
    [ 0.5,           0.0,  np.sqrt(3)/2, 0.2204],
    [ 0.0,           0.0,  0.0,  1.0]
])


# =============================================================================
# Transformation Tree Data Structure
# =============================================================================

class TransformationNode:
    """Represents a coordinate frame in the transformation tree."""

    def __init__(self, name: str, full_name: str, description: str, color: str):
        self.name = name
        self.full_name = full_name
        self.description = description
        self.color = color
        self.children = []
        self.parent = None
        self.transform_to_parent = None
        self.transform_name = None


class TransformationTree:
    """Manages the transformation tree structure."""

    def __init__(self):
        self.nodes: Dict[str, TransformationNode] = {}
        self.root = None

    def add_node(self, name: str, full_name: str, description: str, color: str) -> TransformationNode:
        node = TransformationNode(name, full_name, description, color)
        self.nodes[name] = node
        return node

    def add_edge(self, parent_name: str, child_name: str,
                 transform: Optional[np.ndarray], transform_name: str):
        parent = self.nodes[parent_name]
        child = self.nodes[child_name]
        child.parent = parent
        child.transform_to_parent = transform
        child.transform_name = transform_name
        parent.children.append(child)

    def set_root(self, name: str):
        self.root = self.nodes[name]


def build_transformation_tree() -> TransformationTree:
    """Build the transformation tree from eval.py structure."""

    tree = TransformationTree()

    # Add nodes (coordinate frames)
    tree.add_node(
        "W", "MoCap World",
        "Global reference frame\nfrom Qualisys MoCap system",
        "#3498db"  # Blue
    )

    tree.add_node(
        "S_m", "Swift MoCap",
        "Swift robot pose\nin MoCap world",
        "#e74c3c"  # Red
    )

    tree.add_node(
        "O_m", "Ouster MoCap",
        "Ouster station pose\nin MoCap world",
        "#2ecc71"  # Green
    )

    tree.add_node(
        "S_b", "Swift Body",
        "Swift robot body frame\n(with MOCAP_OFFSET applied)",
        "#e67e22"  # Orange
    )

    tree.add_node(
        "S_fl", "Swift Front LiDAR",
        "Front LiDAR sensor\non Swift robot",
        "#9b59b6"  # Purple
    )

    tree.add_node(
        "S_bl", "Swift Back LiDAR",
        "Back LiDAR sensor\non Swift robot",
        "#9b59b6"  # Purple
    )

    tree.add_node(
        "O_l", "Ouster LiDAR",
        "Ouster OS1-128\nLiDAR sensor",
        "#1abc9c"  # Teal
    )

    # Add edges (transformations)
    tree.add_edge("W", "S_m", None, "T_W_Sm\n(Qualisys pose)")
    tree.add_edge("W", "O_m", None, "T_W_Om\n(Qualisys pose)")
    tree.add_edge("S_m", "S_b", SWIFT_MOCAP_OFFSET, "SWIFT_MOCAP_OFFSET")
    tree.add_edge("S_b", "S_fl", SWIFT_FRONT_LIDAR_FRAME, "SWIFT_FRONT_LIDAR_FRAME")
    tree.add_edge("S_b", "S_bl", SWIFT_BACK_LIDAR_FRAME, "SWIFT_BACK_LIDAR_FRAME")
    tree.add_edge("O_m", "O_l", np.eye(4), "Identity\n(sensor at MoCap marker)")

    tree.set_root("W")

    return tree


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_transformation_tree(tree: TransformationTree,
                                   output_path: str = None,
                                   show: bool = True) -> None:
    """Visualize the transformation tree as a hierarchical diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Define positions for each node
    positions = {
        "W": (5, 8),
        "S_m": (2, 6),
        "O_m": (8, 6),
        "S_b": (2, 4),
        "S_fl": (0.5, 2),
        "S_bl": (3.5, 2),
        "O_l": (8, 4),
    }

    # Draw edges (arrows)
    def draw_arrow(start, end, label, color='#555555', style='->'):
        ax.annotate(
            '',
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle=style,
                color=color,
                lw=2,
                connectionstyle='arc3,rad=0'
            )
        )
        # Label position (midpoint)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Offset label slightly
        offset_x = 0.3 if start[0] < end[0] else -0.3

        ax.text(mid_x + offset_x, mid_y, label, fontsize=8,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.9))

    # Draw edges
    edges = [
        ("W", "S_m", "T_W_Sm\n(Qualisys)"),
        ("W", "O_m", "T_W_Om\n(Qualisys)"),
        ("S_m", "S_b", "SWIFT_MOCAP\n_OFFSET"),
        ("S_b", "S_fl", "SWIFT_FRONT\n_LIDAR_FRAME"),
        ("S_b", "S_bl", "SWIFT_BACK\n_LIDAR_FRAME"),
        ("O_m", "O_l", "Identity"),
    ]

    for parent, child, label in edges:
        p_pos = positions[parent]
        c_pos = positions[child]
        draw_arrow((p_pos[0], p_pos[1] - 0.5), (c_pos[0], c_pos[1] + 0.5), label)

    # Draw registration arrow (dashed, showing what we're estimating)
    ax.annotate(
        '',
        xy=(7.5, 2),
        xytext=(4, 2),
        arrowprops=dict(
            arrowstyle='<->',
            color='#c0392b',
            lw=3,
            linestyle='dashed',
            connectionstyle='arc3,rad=-0.3'
        )
    )
    ax.text(5.75, 0.8, 'T_gt / T_est\n(Registration)', fontsize=10,
            ha='center', va='center', fontweight='bold',
            color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8',
                     edgecolor='#c0392b', alpha=0.9))

    # Draw nodes
    for name, pos in positions.items():
        node = tree.nodes[name]

        # Draw box
        box = FancyBboxPatch(
            (pos[0] - 0.8, pos[1] - 0.4),
            1.6, 0.8,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=node.color,
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)

        # Draw text
        ax.text(pos[0], pos[1] + 0.1, node.name, fontsize=12, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(pos[0], pos[1] - 0.15, node.full_name, fontsize=8,
                ha='center', va='center', color='white')

    # Add legend/description box
    legend_text = """
Transformation Tree - Point Cloud Registration

Frames:
• W: MoCap World (Qualisys global frame)
• S_m: Swift MoCap pose
• O_m: Ouster MoCap pose
• S_b: Swift Body (after MOCAP_OFFSET)
• S_fl/S_bl: Swift Front/Back LiDAR
• O_l: Ouster LiDAR

Data Flow:
1. Swift LiDAR points start in S_fl/S_bl frames
2. Transformed to S_b via inverse of LIDAR_FRAME
3. Transformed to world via T_swift (MoCap + offset)
4. Ouster points transformed via T_ouster (MoCap)
5. Registration estimates T from Swift → Ouster
6. Compare T_est with T_gt from MoCap
"""

    ax.text(10.5, 5, legend_text, fontsize=9,
            ha='left', va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='black', alpha=0.9))

    plt.title('Transformation Tree: Swift-Ouster Registration Evaluation',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        print(f"Saved transformation tree to {output_path}")

    if show:
        plt.show()


def print_transformation_tree(tree: TransformationTree) -> None:
    """Print the transformation tree as ASCII art."""

    print("\n" + "=" * 80)
    print("TRANSFORMATION TREE - Point Cloud Registration Evaluation")
    print("=" * 80)

    tree_str = """
                            ┌─────────────────────┐
                            │   MoCap World (W)   │
                            │  Qualisys Global    │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────┐                 ┌───────────────────┐
        │  Swift MoCap (Sm) │                 │ Ouster MoCap (Om) │
        │   T_W_Sm (pose)   │                 │   T_W_Om (pose)   │
        └─────────┬─────────┘                 └─────────┬─────────┘
                  │                                     │
                  │ SWIFT_MOCAP_OFFSET                  │ Identity
                  ▼                                     ▼
        ┌───────────────────┐                 ┌───────────────────┐
        │  Swift Body (Sb)  │                 │ Ouster LiDAR (Ol) │
        │  Robot body frame │                 │    OS1-128        │
        └─────────┬─────────┘                 └───────────────────┘
                  │                                     ▲
        ┌─────────┴─────────┐                           │
        │                   │                           │
        ▼                   ▼                           │
┌───────────────┐   ┌───────────────┐                   │
│Swift Front    │   │Swift Back     │                   │
│LiDAR (Sfl)    │   │LiDAR (Sbl)    │                   │
│               │   │               │                   │
└───────┬───────┘   └───────┬───────┘                   │
        │                   │                           │
        └─────────┬─────────┘                           │
                  │                                     │
                  │     ════════════════════════════════╪═══
                  │        T_gt (Ground Truth) /        │
                  └────────► T_est (Estimated)  ────────┘
                           Point Cloud Registration

    """
    print(tree_str)

    print("\n" + "-" * 80)
    print("TRANSFORMATION MATRICES")
    print("-" * 80)

    print("\n1. SWIFT_MOCAP_OFFSET (Swift MoCap → Swift Body):")
    print("   Accounts for offset between MoCap markers and robot body frame")
    print(f"   Translation: [{SWIFT_MOCAP_OFFSET[0,3]:.4f}, {SWIFT_MOCAP_OFFSET[1,3]:.4f}, {SWIFT_MOCAP_OFFSET[2,3]:.4f}] m")

    print("\n2. SWIFT_FRONT_LIDAR_FRAME (Swift Body → Front LiDAR):")
    print(f"   Translation: [{SWIFT_FRONT_LIDAR_FRAME[0,3]:.4f}, {SWIFT_FRONT_LIDAR_FRAME[1,3]:.4f}, {SWIFT_FRONT_LIDAR_FRAME[2,3]:.4f}] m")
    print("   (30° downward tilt, front of robot)")

    print("\n3. SWIFT_BACK_LIDAR_FRAME (Swift Body → Back LiDAR):")
    print(f"   Translation: [{SWIFT_BACK_LIDAR_FRAME[0,3]:.4f}, {SWIFT_BACK_LIDAR_FRAME[1,3]:.4f}, {SWIFT_BACK_LIDAR_FRAME[2,3]:.4f}] m")
    print("   (30° downward tilt, back of robot)")

    print("\n" + "-" * 80)
    print("DATA FLOW IN eval.py")
    print("-" * 80)

    data_flow = """
1. EXTRACT POINT CLOUDS:
   - Swift front LiDAR points (in S_fl frame)
   - Swift back LiDAR points (in S_bl frame)
   - Ouster LiDAR points (in O_l frame)

2. MERGE SWIFT LIDARS (optional):
   - front_body = front @ inv(SWIFT_FRONT_LIDAR_FRAME)  → S_b frame
   - back_body = back @ inv(SWIFT_BACK_LIDAR_FRAME)    → S_b frame
   - merged = concatenate(front_body, back_body)

3. TRANSFORM TO WORLD FRAME:
   - T_swift = pose_to_matrix(MoCap_swift) + SWIFT_MOCAP_OFFSET
   - T_ouster = pose_to_matrix(MoCap_ouster)
   - src = transform(swift_points, T_swift)  → World frame
   - tgt = transform(ouster_points, T_ouster) → World frame

4. COMPUTE GROUND TRUTH:
   - T_gt = relative_transform(MoCap_swift, MoCap_ouster)

5. RUN REGISTRATION:
   - T_est = registration_algorithm(src, tgt)

6. EVALUATE:
   - rotation_error = angle(T_gt.R, T_est.R)
   - translation_error = ||T_gt.t - T_est.t||
   - translation_magnitude_error = |T_est.t| - |T_gt.t|
"""
    print(data_flow)

    print("=" * 80)


def visualize_transformation_matrices() -> None:
    """Visualize the actual transformation matrices."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    matrices = [
        ("SWIFT_MOCAP_OFFSET", SWIFT_MOCAP_OFFSET),
        ("SWIFT_FRONT_LIDAR_FRAME", SWIFT_FRONT_LIDAR_FRAME),
        ("SWIFT_BACK_LIDAR_FRAME", SWIFT_BACK_LIDAR_FRAME),
    ]

    for ax, (name, matrix) in zip(axes, matrices):
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1)

        # Add text annotations
        for i in range(4):
            for j in range(4):
                val = matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color=color, fontsize=9)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(['R00/t0', 'R01/t1', 'R02/t2', 'tx/ty/tz/1'])
        ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'])
        ax.set_title(name, fontweight='bold', fontsize=10)

    plt.colorbar(im, ax=axes, shrink=0.8, label='Value')
    plt.suptitle('Transformation Matrices from eval.py', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/aaron/PycharmProjects/PythonProject/thesis_resources/transformation_matrices.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Build and visualize the transformation tree
    tree = build_transformation_tree()

    # Print ASCII tree
    print_transformation_tree(tree)

    # Create visual diagram
    visualize_transformation_tree(
        tree,
        output_path='/home/aaron/PycharmProjects/PythonProject/thesis_resources/transformation_tree.png',
        show=True
    )

    # Visualize matrices
    visualize_transformation_matrices()
