"""
Configuration file for Air Canvas application.

Modify these settings to customize the application behavior.
"""

# Camera Settings
CAMERA_INDEX = 0  # Primary camera (change to 1, 2, etc. for different cameras)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Hand Detection Settings
DETECTION_CONFIDENCE = 0.7  # 0.0 to 1.0 - higher = stricter detection
TRACKING_CONFIDENCE = 0.5   # 0.0 to 1.0 - higher = stricter tracking

# Drawing Settings
DEFAULT_COLOR = "green"  # Starting color
DEFAULT_THICKNESS = 5  # Starting brush thickness in pixels

# Color Palette (BGR format for OpenCV)
COLOR_PALETTE = {
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# Brush Thickness Options
THICKNESS_OPTIONS = [3, 5, 8, 12, 15]

# Gesture Recognition
GESTURE_THRESHOLD = 50  # Pixel threshold for gesture detection

# UI Settings
SHOW_FPS = True
SHOW_HAND_LANDMARKS = False  # Draw hand landmarks on screen
CANVAS_OPACITY = 0.7  # 0.0 to 1.0 - transparency of canvas overlay

# File Settings
SAVE_DIRECTORY = "./drawings"  # Directory to save drawings
SAVE_FORMAT = "png"  # Image format: 'png', 'jpg', etc.

# Performance Settings
ENABLE_SMOOTHING = True  # Smooth drawing trajectories
SMOOTHING_BUFFER_SIZE = 2  # Number of points for smoothing

# Debug Settings
DEBUG_MODE = False  # Print detailed debug information
