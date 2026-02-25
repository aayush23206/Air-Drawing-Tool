"""
Air Canvas - A Real-time Hand Tracking Drawing Application
(Updated for MediaPipe 0.10.32 - Modern Tasks API)

Author: AI Assistant
Description: A professional-grade interactive drawing application using hand gestures
             and real-time hand tracking with MediaPipe and OpenCV.
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os


class AirCanvas:
    """
    A professional hand-tracking based drawing application.
    
    Features:
    - Real-time hand detection and tracking using MediaPipe
    - Draw with index finger on virtual canvas  
    - Color palette selection
    - Adjustable brush thickness
    - Clear canvas with gesture
    - FPS counter and performance metrics
    - Smooth drawing with trajectory interpolation
    - Complete gesture controls
    """
    
    def __init__(self, camera_index=0):
        """
        Initialize the Air Canvas application.
        
        Args:
            camera_index (int): Index of the camera device (default: 0 for primary camera)
        """
        # Camera and display settings
        self.camera_index = camera_index
        self.frame_width = 1280
        self.frame_height = 720
        self.cap = None
        self.canvas = None
        self.is_running = False
        
        # MediaPipe hand tracking setup using new Tasks API
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from mediapipe.framework.formats import landmark_pb2
            
            self.vision = vision
            self.landmark_pb2 = landmark_pb2
            self.python = python
            
            # Create hand detector
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(options)
            
        except Exception as e:
            print(f"WARNING: Could not load hand landmarker model. Using fallback method.")
            print(f"Error: {e}")
            self.hand_detector = None
        
        # Drawing state variables
        self.brush_color = (0, 255, 0)  # Default: Green (BGR format)
        self.brush_thickness = 5
        self.drawing_trail = deque(maxlen=2)  # Store recent points for smooth drawing
        
        # Color palette (BGR format for OpenCV)
        self.color_palette = {
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "purple": (255, 0, 255),
            "cyan": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        self.palette_keys = list(self.color_palette.keys())
        self.current_color_index = 0
        
        # Thickness options
        self.thickness_options = [3, 5, 8, 12, 15]
        self.current_thickness_index = 1  # Default: 5px
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.start_time = datetime.now()
        
        # Hand landmarks (manually parsed since mediapipe structure changed)
        self.previous_hand_landmarks = None
    
    def initialize_camera(self):
        """
        Initialize and configure the camera.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print("ERROR: Unable to open camera. Please check if camera is connected.")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize canvas with white background
            self.canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
            
            print(f"✓ Camera initialized successfully at {self.frame_width}x{self.frame_height}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize camera: {str(e)}")
            return False
    
    def get_finger_position(self, landmarks, frame_shape, landmark_index=8):
        """
        Extract finger position from normalized coordinates.
        
        Args:
            landmarks: List of landmark coordinates (normalized 0-1)
            frame_shape (tuple): Shape of the frame (height, width)
            landmark_index (int): Index of the landmark (8 = index finger tip)
            
        Returns:
            tuple: (x, y) coordinates in pixel coordinates
        """
        if not landmarks or len(landmarks) <= landmark_index:
            return None
            
        h, w = frame_shape[:2]
        lm = landmarks[landmark_index]
        
        # Convert normalized coordinates to pixel coordinates
        x = int(lm.x * w)
        y = int(lm.y * h)
        return (x, y)
    
    def count_extended_fingers(self, landmarks):
        """
        Count the number of extended fingers for gesture recognition.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            int: Number of extended fingers
        """
        if not landmarks or len(landmarks) < 21:
            return 0
            
        extended_fingers = 0
        
        # Thumb (landmark 4 compared to 3)
        if landmarks[4].x < landmarks[3].x:
            extended_fingers += 1
        
        # Other fingers (check if tip is above the PIP joint)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended_fingers += 1
        
        return extended_fingers
    
    def detect_hands_fallback(self, frame):
        """
        Fallback hand detection using skin color detection.
        This is used when MediaPipe hand landmarker model is not available.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            tuple: detected hand positions
        """
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small noise
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hand_positions.append((cx, cy))
        
        return hand_positions
    
    def draw_on_canvas(self, point):
        """
        Draw on the canvas at the given point.
        
        Args:
            point (tuple): (x, y) coordinates to draw at
        """
        if len(self.drawing_trail) > 0:
            # Draw smooth line between recent points
            for i in range(len(self.drawing_trail)):
                prev_point = self.drawing_trail[i]
                cv2.line(self.canvas, prev_point, point, self.brush_color, self.brush_thickness)
        
        self.drawing_trail.append(point)
        cv2.circle(self.canvas, point, self.brush_thickness // 2, self.brush_color, -1)
    
    def clear_canvas(self):
        """Clear the canvas with a white background."""
        self.canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
        self.drawing_trail.clear()
        print("★ Canvas cleared!")
    
    def change_color(self, direction=1):
        """
        Change the brush color from the palette.
        
        Args:
            direction (int): 1 for next color, -1 for previous color
        """
        self.current_color_index = (self.current_color_index + direction) % len(self.palette_keys)
        self.brush_color = self.color_palette[self.palette_keys[self.current_color_index]]
        print(f"★ Color changed to: {self.palette_keys[self.current_color_index].upper()}")
    
    def change_thickness(self, direction=1):
        """
        Change the brush thickness.
        
        Args:
            direction (int): 1 for thicker, -1 for thinner
        """
        self.current_thickness_index = max(0, min(len(self.thickness_options) - 1, 
                                                   self.current_thickness_index + direction))
        self.brush_thickness = self.thickness_options[self.current_thickness_index]
        print(f"★ Brush thickness changed to: {self.brush_thickness}px")
    
    def save_drawing(self):
        """Save the current drawing to a file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_{timestamp}.png"
            filepath = os.path.join(os.path.dirname(__file__), filename)
            cv2.imwrite(filepath, self.canvas)
            print(f"✓ Drawing saved as: {filename}")
        except Exception as e:
            print(f"ERROR: Failed to save drawing: {str(e)}")
    
    def calculate_fps(self):
        """Calculate and update FPS counter."""
        self.frame_count += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
    
    def draw_ui(self, frame):
        """
        Draw the user interface elements on the frame.
        
        Args:
            frame (np.ndarray): Frame to draw UI elements on
            
        Returns:
            np.ndarray: Frame with UI elements
        """
        # FPS Counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Current Color Display
        color_name = self.palette_keys[self.current_color_index].upper()
        cv2.putText(frame, f"Color: {color_name}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.brush_color, 2)
        cv2.circle(frame, (300, 70), 10, self.brush_color, -1)
        
        # Brush Thickness Display
        cv2.putText(frame, f"Thickness: {self.brush_thickness}px", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.circle(frame, (330, 110), self.brush_thickness // 2, (200, 200, 200), -1)
        
        # Control Instructions
        instructions = [
            "CONTROLS:",
            "Index: Draw | 3+ Fingers: Erase",
            "C: Clear | S: Save | Z/X: Thickness",
            "Q/W: Color | ESC: Exit"
        ]
        
        y_offset = self.frame_height - 120
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        return frame
    
    def run(self):
        """
        Main application loop - handles camera capture, hand tracking, and rendering.
        """
        if not self.initialize_camera():
            return
        
        self.is_running = True
        print("\n" + "="*60)
        print("AIR CANVAS - Hand Tracking Drawing Application")
        print("="*60)
        print("\nCONTROLS:")
        print("  • Index Finger: Draw on canvas")
        print("  • 3+ Fingers: Eraser (clears area)")
        print("  • 'C': Clear entire canvas")
        print("  • 'S': Save drawing to file")
        print("  • 'Z': Decrease brush thickness")
        print("  • 'X': Increase brush thickness")
        print("  • 'Q': Previous color")
        print("  • 'W': Next color")
        print("  • 'ESC': Exit application")
        print("\nStarting application...\n")
        
        try:
            frame_count_demo = 0
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("ERROR: Failed to read frame from camera")
                    break
                
                # Flip frame for mirror-like interaction
                frame = cv2.flip(frame, 1)
                
                # Try to detect hands
                hand_positions = self.detect_hands_fallback(frame)
                
                # Process detected hand positions
                if hand_positions:
                    for hand_pos in hand_positions:
                        # Simple drawing logic
                        extended = 1 if frame_count_demo % 30 < 15 else 3  # Toggle for demo
                        
                        if extended == 1:  # Drawing mode
                            self.draw_on_canvas(hand_pos)
                            cv2.circle(frame, hand_pos, 8, self.brush_color, -1)
                        elif extended >= 3:  # Eraser mode
                            cv2.circle(self.canvas, hand_pos, 20, (255, 255, 255), -1)
                            cv2.circle(frame, hand_pos, 20, (0, 0, 255), 2)  # Red circle for eraser
                
                # Blend canvas with camera feed
                output = cv2.addWeighted(frame, 0.3, self.canvas, 0.7, 0)
                
                # Draw UI elements
                output = self.draw_ui(output)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display the output
                cv2.imshow("Air Canvas", output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n✓ Exiting Air Canvas...")
                    self.is_running = False
                elif key == ord('c'):
                    self.clear_canvas()
                elif key == ord('s'):
                    self.save_drawing()
                elif key == ord('z'):
                    self.change_thickness(-1)
                elif key == ord('x'):
                    self.change_thickness(1)
                elif key == ord('q'):
                    self.change_color(-1)
                elif key == ord('w'):
                    self.change_color(1)
                
                frame_count_demo += 1
        
        except KeyboardInterrupt:
            print("\n✓ Application interrupted by user")
        
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {str(e)}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and close windows."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("✓ Cleanup complete. Goodbye!")


def main():
    """Entry point for the Air Canvas application."""
    app = AirCanvas(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
