# ============================================================
# config_variables.py
# All tweakable numbers in one place so you can experiment easily.
# Change any value here and re-run main.py to see the effect.
# ============================================================

# --- Screen and Display ---
SCREEN_WIDTH = 1400          # Total window width in pixels
SCREEN_HEIGHT = 900          # Total window height in pixels
TOP_PANEL_HEIGHT = 240       # Height of the neural-network + stats panel at the top
FPS = 60                     # Target frames per second

# --- Car Physics ---
MAX_SPEED = 8.0              # Maximum speed a car can reach (pixels per frame)
ACCELERATION = 0.25          # Speed gained per frame when accelerating
BRAKE_POWER = 0.4            # Speed lost per frame when braking
TURN_SPEED = 3.5             # Max turning in degrees per frame
FRICTION = 0.05              # Speed lost every frame automatically (drag)
CAR_LENGTH = 30              # Car body length in pixels
CAR_WIDTH = 16               # Car body width in pixels

# --- Sensors ---
SENSOR_COUNT = 8             # Number of distance sensors around the car
SENSOR_LENGTH = 200          # Max distance a sensor ray can reach (pixels)
SENSOR_STEP = 5              # Step size when walking along a sensor ray (pixels)

# --- Road Generation ---
ROAD_WIDTH = 100             # Total width of the asphalt road (pixels)
ROAD_CONTROL_POINTS = 80     # Number of random waypoints that define the road shape
ROAD_POINT_SPACING = 300     # Vertical gap between consecutive control points (pixels)
ROAD_WIGGLE = 250            # Max random horizontal offset of each control point
ROAD_SMOOTH_POINTS = 25      # Interpolated points between each pair of control points
ROAD_SEED = 42               # Fixed random seed so each generation drives the same road
                              # Set to None if you want a new road every generation

# --- Colors ---
COLOR_GRASS = (46, 139, 87)           # Sea-green grass
COLOR_ASPHALT = (55, 55, 60)          # Dark asphalt
COLOR_CENTER_LINE = (255, 255, 255)   # White dashed center line
COLOR_CURB_RED = (200, 40, 40)        # Red part of the curb
COLOR_CURB_WHITE = (235, 235, 235)    # White part of the curb
COLOR_TREE_DARK = (25, 100, 30)       # Dark green tree circles
COLOR_TREE_LIGHT = (50, 140, 50)      # Light green tree circles

COLOR_CAR_BODY = (40, 140, 255)       # Blue car body
COLOR_CAR_BEST = (255, 210, 0)        # Gold highlight for the best car
COLOR_SENSOR_LINE = (255, 255, 80)    # Yellow sensor rays
COLOR_SENSOR_DOT = (255, 50, 50)      # Red dot at the end of each sensor ray

COLOR_PANEL_BG = (20, 22, 30)         # Very dark blue-grey panel background
COLOR_PANEL_BORDER = (60, 65, 80)     # Subtle border for the panel
COLOR_TEXT = (210, 215, 225)           # Light text
COLOR_TEXT_DIM = (130, 135, 145)       # Dimmed text
COLOR_POSITIVE_WEIGHT = (50, 210, 80) # Green for positive connection weights
COLOR_NEGATIVE_WEIGHT = (220, 60, 60) # Red for negative connection weights
COLOR_NODE_RING = (180, 185, 200)     # Node outline color

# --- Simulation ---
MAX_FRAMES_PER_GEN = 1800             # End the generation after this many frames
STUCK_TIMEOUT_SECONDS = 4             # Kill a car if no forward progress for this long
STUCK_TIMEOUT_FRAMES = STUCK_TIMEOUT_SECONDS * FPS  # Converted to frame count
BACKWARD_THRESHOLD = 15               # Kill car if it goes this many road-points backward

# --- Fitness ---
SPEED_BONUS_WEIGHT = 0.1              # Tiny bonus multiplied by current speed each frame

# --- NEAT ---
POPULATION_SIZE = 50                  # Must match pop_size in config_file.txt
