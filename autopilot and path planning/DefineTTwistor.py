"""
DefineTTwistor - Python class equivalent of the MATLAB DefineTTwistor.m script.

Defines the vertices and wireframe point arrays needed to animate the TTwistor
aircraft in AnimateSimulation. Also defines face/color data (currently unused
in the animation, but included for completeness).

Usage
-----
aircraft = DefineTTwistor()
pts  = aircraft.pts    # dict with keys: 'fuse', 'wing', 'tailwing', 'tail'
                       # each value is an ndarray of shape (3, n_pts)
F    = aircraft.F      # face index array (6, 4)
colors = aircraft.patch_colors  # (6, 3) RGB array
"""

import numpy as np
from dataclasses import dataclass, field


class DefineTTwistor:
    """
    Defines the TTwistor aircraft geometry for use in 3-D animation.

    Attributes
    ----------
    pts : dict
        Wireframe point arrays (shape 3 × n) for each aircraft component:
        'fuse', 'wing', 'tailwing', 'tail'.
    V : ndarray, shape (19, 3)
        Raw vertex positions in the body NED frame.
    F : ndarray, shape (6, 4)
        Face definitions (1-based indices preserved as 0-based integers).
    patch_colors : ndarray, shape (6, 3)
        RGB colour for each face.
    """

    def __init__(self):
        self._define_geometry()

    def _define_geometry(self):
        # ----------------------------------------------------------------
        # Geometry parameters
        # ----------------------------------------------------------------
        fuse_h  = 1.0
        fuse_w  = 1.0
        fuse_l1 = 3.0
        fuse_l2 = 1.0
        fuse_l3 = 8.0    # original comment: was 10

        wing_l  = 1.5
        wing_w  = 10.5   # original comment: was 9

        tailwing_l = 0.8  # original comment: was 1
        tailwing_w = 4.0

        tail_h = 2.0      # original comment: was 3

        # ----------------------------------------------------------------
        # Vertices  (NED body frame, 0-indexed in Python)
        # ----------------------------------------------------------------
        V = np.array([
            [ fuse_l1,              0,              0          ],  # 0  (pt 1)
            [ fuse_l2,  fuse_w / 2,     -fuse_h / 2            ],  # 1  (pt 2)
            [ fuse_l2, -fuse_w / 2,     -fuse_h / 2            ],  # 2  (pt 3)
            [ fuse_l2, -fuse_w / 2,      fuse_h / 2            ],  # 3  (pt 4)
            [ fuse_l2,  fuse_w / 2,      fuse_h / 2            ],  # 4  (pt 5)
            [-fuse_l3,              0,              0          ],  # 5  (pt 6)

            [ 0,        wing_w / 2,              0             ],  # 6  (pt 7)
            [-0.44,     wing_w / 2 + 1.06,       0             ],  # 7  (pt 8)
            [-wing_l,   wing_w / 2 + 1.5,        0             ],  # 8  (pt 9)
            [-wing_l,  -wing_w / 2 - 1.5,        0             ],  # 9  (pt 10)
            [-0.44,    -wing_w / 2 - 1.06,       0             ],  # 10 (pt 11)
            [ 0,       -wing_w / 2,              0             ],  # 11 (pt 12)

            [-fuse_l3 + tailwing_l,  tailwing_w / 2, -tail_h  ],  # 12 (pt 13)
            [-fuse_l3,               tailwing_w / 2, -tail_h  ],  # 13 (pt 14)
            [-fuse_l3,              -tailwing_w / 2, -tail_h  ],  # 14 (pt 15)
            [-fuse_l3 + tailwing_l, -tailwing_w / 2, -tail_h  ],  # 15 (pt 16)

            [-fuse_l3 + 1.5 * tailwing_l,  0,  0              ],  # 16 (pt 17)
            [-fuse_l3 + tailwing_l,         0, -tail_h        ],  # 17 (pt 18)
            [-fuse_l3,                      0, -tail_h        ],  # 18 (pt 19)
        ])

        self.V = V

        # ----------------------------------------------------------------
        # Wireframe point arrays  (shape: 3 × n_pts)
        # MATLAB indices converted to 0-based
        # ----------------------------------------------------------------
        pts = {}

        pts['fuse'] = V[[0, 1, 5, 2, 0, 4, 5, 3, 0]].T
        #              pt1, pt2, pt6, pt3, pt1, pt5, pt6, pt4, pt1

        pts['wing'] = V[[6, 7, 8, 9, 10, 11, 6]].T
        #              pt7, pt8, pt9, pt10, pt11, pt12, pt7

        pts['tailwing'] = V[[12, 13, 14, 15, 12]].T
        #                  pt13, pt14, pt15, pt16, pt13

        pts['tail'] = V[[5, 16, 17, 18, 5]].T
        #              pt6, pt17, pt18, pt19, pt6

        self.pts = pts

        # ----------------------------------------------------------------
        # Face definitions  (0-based indices, shape 6 × 4)
        # ---- NOT currently used by the animator ----
        # ----------------------------------------------------------------
        self.F = np.array([
            [0, 1,  5,  4],   # front
            [3, 2,  6,  7],   # back
            [0, 4,  7,  3],   # right
            [1, 5,  6,  2],   # left
            [4, 5,  6,  7],   # top
            [8, 9, 10, 11],   # bottom
        ], dtype=int)

        # ----------------------------------------------------------------
        # Face colours  (RGB, shape 6 × 3)
        # ---- NOT currently used by the animator ----
        # ----------------------------------------------------------------
        self.patch_colors = np.array([
            [1, 0, 0],   # front  - red
            [0, 1, 0],   # back   - green
            [0, 0, 1],   # right  - blue
            [1, 1, 0],   # left   - yellow
            [0, 1, 1],   # top    - cyan
            [0, 1, 1],   # bottom - cyan
        ], dtype=float)


# ---------------------------------------------------------------------------
# Quick visual check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ac = DefineTTwistor()

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    colors = {'fuse': 'k', 'wing': 'b', 'tailwing': 'r', 'tail': 'g'}
    for name, pts in ac.pts.items():
        ax.plot(pts[1], pts[0], -pts[2], color=colors[name], label=name)

    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Alt (-Down)')
    ax.set_title('TTwistor geometry check')
    ax.legend()
    plt.tight_layout()
    plt.show()