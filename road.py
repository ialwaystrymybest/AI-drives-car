# ============================================================
# road.py — Procedural road generation
# Creates a smooth winding road using random control points
# and scipy CubicSpline. Also handles collision checking via
# a precomputed grid for O(1) "is this point on the road?" lookups.
# ============================================================

import math
import numpy as np
from scipy.interpolate import CubicSpline
import pygame
from config_variables import *

# How many pixels each grid cell covers (smaller = more accurate but slower to build)
GRID_CELL_SIZE = 5


class Road:
    """Holds the road center-line, boundaries, and provides collision checks."""

    def __init__(self, seed=42):
        # ----------------------------------------------------------
        # Step 1: Generate random control points for the road shape
        # The road starts at the bottom and goes upward (decreasing y).
        # ----------------------------------------------------------
        rng = np.random.RandomState(seed)

        num_pts = ROAD_CONTROL_POINTS
        ctrl_x = [SCREEN_WIDTH // 2]            # Start in the middle of the screen
        ctrl_y = [0.0]                           # Start at world y = 0

        for i in range(1, num_pts):
            # Each point moves upward (negative y) with a random horizontal wiggle
            new_x = ctrl_x[-1] + rng.randint(-ROAD_WIGGLE, ROAD_WIGGLE + 1)
            new_y = ctrl_y[-1] - ROAD_POINT_SPACING   # Go upward
            ctrl_x.append(float(new_x))
            ctrl_y.append(float(new_y))

        # ----------------------------------------------------------
        # Step 2: Smooth the control points with CubicSpline
        # We parameterize by t = 0, 1, 2, … and get smooth x(t), y(t).
        # ----------------------------------------------------------
        t_ctrl = np.arange(num_pts, dtype=float)
        cs_x = CubicSpline(t_ctrl, ctrl_x)
        cs_y = CubicSpline(t_ctrl, ctrl_y)

        total_points = (num_pts - 1) * ROAD_SMOOTH_POINTS + 1
        t_smooth = np.linspace(0, num_pts - 1, total_points)

        self.center_x = cs_x(t_smooth)          # Numpy array of center-line x coords
        self.center_y = cs_y(t_smooth)           # Numpy array of center-line y coords
        self.num_points = len(self.center_x)
        self.half_width = ROAD_WIDTH / 2.0
        self.half_width_sq = self.half_width ** 2

        # ----------------------------------------------------------
        # Step 3: Compute tangent and normal at each center-line point
        # The normal is perpendicular to the road direction so we can
        # offset left and right to build road boundaries.
        # ----------------------------------------------------------
        tangent_x = np.gradient(self.center_x)
        tangent_y = np.gradient(self.center_y)
        tangent_len = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
        tangent_len = np.maximum(tangent_len, 0.001)  # Avoid divide-by-zero

        # Normal pointing to the RIGHT of the driving direction
        self.normal_x = -tangent_y / tangent_len
        self.normal_y = tangent_x / tangent_len

        # ----------------------------------------------------------
        # Step 4: Build left and right road boundaries
        # ----------------------------------------------------------
        self.left_x = self.center_x - self.normal_x * self.half_width
        self.left_y = self.center_y - self.normal_y * self.half_width
        self.right_x = self.center_x + self.normal_x * self.half_width
        self.right_y = self.center_y + self.normal_y * self.half_width

        # ----------------------------------------------------------
        # Step 5: Cumulative distance along the center-line (for fitness)
        # ----------------------------------------------------------
        dx = np.diff(self.center_x)
        dy = np.diff(self.center_y)
        segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
        self.cumulative_dist = np.zeros(self.num_points)
        self.cumulative_dist[1:] = np.cumsum(segment_lengths)

        # ----------------------------------------------------------
        # Step 6: Starting position and angle for the cars
        # ----------------------------------------------------------
        start_i = 10  # A few points in so the car is clearly on the road
        self.start_x = float(self.center_x[start_i])
        self.start_y = float(self.center_y[start_i])

        # Angle the car should face (our convention: 0 = north, clockwise positive)
        sdx = self.center_x[start_i + 1] - self.center_x[start_i - 1]
        sdy = self.center_y[start_i + 1] - self.center_y[start_i - 1]
        self.start_angle = math.degrees(math.atan2(sdx, -sdy))

        # ----------------------------------------------------------
        # Step 7: Build a grid for fast "is on road?" checks
        # Each cell is True if it is inside the road surface.
        # ----------------------------------------------------------
        self._build_road_grid()

        # ----------------------------------------------------------
        # Step 8: Generate decorative trees beside the road
        # ----------------------------------------------------------
        self.trees = []
        tree_rng = np.random.RandomState((seed + 7) if seed else None)
        for i in range(0, self.num_points, 20):
            for side in [-1, 1]:
                if tree_rng.random() < 0.55:
                    offset = self.half_width + tree_rng.randint(35, 130)
                    tx = self.center_x[i] + self.normal_x[i] * offset * side
                    ty = self.center_y[i] + self.normal_y[i] * offset * side
                    radius = tree_rng.randint(12, 28)
                    self.trees.append((float(tx), float(ty), int(radius)))

    # ==============================================================
    # Grid builder — marks every cell that is inside the road.
    # Uses a precomputed circle mask stamped onto the grid with fast
    # array slices instead of 1+ million slow nested Python loop steps.
    # ==============================================================
    def _build_road_grid(self):
        all_x = np.concatenate([self.left_x, self.right_x])
        all_y = np.concatenate([self.left_y, self.right_y])
        self._grid_x0 = float(all_x.min()) - 50
        self._grid_y0 = float(all_y.min()) - 50
        grid_w = int((all_x.max() - self._grid_x0 + 100) / GRID_CELL_SIZE) + 1
        grid_h = int((all_y.max() - self._grid_y0 + 100) / GRID_CELL_SIZE) + 1
        self._grid = np.zeros((grid_h, grid_w), dtype=np.bool_)
        self._grid_w = grid_w
        self._grid_h = grid_h

        hw_cells = int(self.half_width / GRID_CELL_SIZE) + 1
        hw_sq = hw_cells * hw_cells

        # Precompute the boolean circle mask once
        y, x = np.ogrid[-hw_cells:hw_cells + 1, -hw_cells:hw_cells + 1]
        circle_mask = (x * x + y * y) <= hw_sq

        # Pre-convert all center points to grid coordinates
        cxs = ((self.center_x - self._grid_x0) / GRID_CELL_SIZE).astype(int)
        cys = ((self.center_y - self._grid_y0) / GRID_CELL_SIZE).astype(int)

        # Stamp the circle mask onto the grid for every center point
        for i in range(self.num_points):
            cx = cxs[i]
            cy = cys[i]

            # Grid slice boundaries with clipping to grid edges
            g_y0 = max(0, cy - hw_cells)
            g_y1 = min(self._grid_h, cy + hw_cells + 1)
            g_x0 = max(0, cx - hw_cells)
            g_x1 = min(self._grid_w, cx + hw_cells + 1)

            # Corresponding slice in the precomputed circle mask
            m_y0 = g_y0 - (cy - hw_cells)
            m_y1 = m_y0 + (g_y1 - g_y0)
            m_x0 = g_x0 - (cx - hw_cells)
            m_x1 = m_x0 + (g_x1 - g_x0)

            self._grid[g_y0:g_y1, g_x0:g_x1] |= circle_mask[m_y0:m_y1, m_x0:m_x1]

    # ==============================================================
    # Check if a world-coordinate point is on the road  (O(1) lookup)
    # ==============================================================
    def is_on_road(self, x, y):
        gx = int((x - self._grid_x0) / GRID_CELL_SIZE)
        gy = int((y - self._grid_y0) / GRID_CELL_SIZE)
        if 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
            return bool(self._grid[gy, gx])
        return False

    # ==============================================================
    # Find the index of the nearest center-line point to (x, y).
    # hint_index speeds up the search by only looking nearby.
    # ==============================================================
    def get_nearest_index(self, x, y, hint_index=0):
        s = max(0, hint_index - 30)
        e = min(self.num_points, hint_index + 60)
        dx = self.center_x[s:e] - x
        dy = self.center_y[s:e] - y
        dist_sq = dx * dx + dy * dy
        return int(s + np.argmin(dist_sq))

    # ==============================================================
    # Draw the visible portion of the road
    # ==============================================================
    def draw(self, screen, camera_x, camera_y):
        # Find which center-line points are currently on-screen
        screen_y = self.center_y - camera_y + TOP_PANEL_HEIGHT
        visible = np.where((screen_y > -300) & (screen_y < SCREEN_HEIGHT + 300))[0]
        if len(visible) == 0:
            return
        si = max(0, int(visible[0]) - 1)                     # start index
        ei = min(self.num_points - 1, int(visible[-1]) + 1)   # end index

        # Helper: convert world coords to screen coords
        def sx(wx):
            return int(wx - camera_x)
        def sy(wy):
            return int(wy - camera_y + TOP_PANEL_HEIGHT)

        # --- Draw road surface as one filled polygon ---
        poly = []
        for i in range(si, ei + 1):
            poly.append((sx(self.left_x[i]), sy(self.left_y[i])))
        for i in range(ei, si - 1, -1):
            poly.append((sx(self.right_x[i]), sy(self.right_y[i])))
        if len(poly) >= 3:
            pygame.draw.polygon(screen, COLOR_ASPHALT, poly)

        # --- Draw curbs (alternating red and white) ---
        for i in range(si, ei):
            color = COLOR_CURB_RED if (i // 3) % 2 == 0 else COLOR_CURB_WHITE
            pygame.draw.line(screen, color,
                             (sx(self.left_x[i]), sy(self.left_y[i])),
                             (sx(self.left_x[i+1]), sy(self.left_y[i+1])), 5)
            pygame.draw.line(screen, color,
                             (sx(self.right_x[i]), sy(self.right_y[i])),
                             (sx(self.right_x[i+1]), sy(self.right_y[i+1])), 5)

        # --- Draw dashed white center-line ---
        for i in range(si, ei):
            if (i % 4) < 2:  # Draw 2 segments, skip 2 segments
                pygame.draw.line(screen, COLOR_CENTER_LINE,
                                 (sx(self.center_x[i]), sy(self.center_y[i])),
                                 (sx(self.center_x[i+1]), sy(self.center_y[i+1])), 2)

        # --- Draw trees ---
        for tx, ty, r in self.trees:
            tsx = sx(tx)
            tsy = sy(ty)
            if -40 < tsx < SCREEN_WIDTH + 40 and -40 < tsy < SCREEN_HEIGHT + 40:
                # Dark trunk shadow then lighter canopy
                pygame.draw.circle(screen, COLOR_TREE_DARK, (tsx, tsy), r)
                pygame.draw.circle(screen, COLOR_TREE_LIGHT, (tsx - 2, tsy - 2), r - 3)
