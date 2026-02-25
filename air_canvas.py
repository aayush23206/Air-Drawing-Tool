"""
Air Canvas - Professional Hand Tracking Drawing Application
(OpenCV-based hand detection - compatible with all systems)

Author: AI Assistant
Description: A production-ready interactive drawing application using hand gesture
             recognition and real-time color-based hand tracking with OpenCV.
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
    - Real-time hand detection using skin color segmentation
    - Draw with index finger on virtual canvas  
    - Color palette selection (8 colors)
    - Adjustable brush thickness (5 levels)
    - Clear canvas gesture
    - FPS counter and performance metrics
    - Smooth drawing with trajectory interpolation
    - Complete gesture controls and error handling
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
        
        # Skin detection calibration
        self.skin_lower = np.array([0, 15, 60], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.calibrated = False
        
        # Drawing state variables
        self.brush_color = (0, 255, 0)  # Default: Green (BGR format)
        self.brush_thickness = 5
        self.drawing_trail = deque(maxlen=3)  # Store recent points for smooth drawing
        self.is_drawing = False
        
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
        self.last_hand_center = None
    
    def initialize_camera(self):
        """
        Initialize and configure the camera.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print("❌ ERROR: Unable to open camera. Please check if camera is connected.")
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
            print(f"❌ ERROR: Failed to initialize camera: {str(e)}")
            return False
    
    def calibrate_skin_color(self, frame):
        """
        Calibrate skin color detection from hand in center ROI.
        Improved version with better sampling.
        
        Args:
            frame (np.ndarray): Input video frame
        """
        # Define ROI in center of frame
        h, w = frame.shape[:2]
        roi_size = 150  # Larger ROI for better sampling
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Convert to HSV and get statistics
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Get average skin tone
        avg_h = np.median(hsv[:,:,0])
        avg_s = np.median(hsv[:,:,1])
        avg_v = np.median(hsv[:,:,2])
        
        # Create adaptive thresholds with larger margins for robustness
        margin_h = 15
        margin_s = 40
        margin_v = 60
        
        self.skin_lower = np.array([
            max(0, avg_h - margin_h),
            max(0, avg_s - margin_s),
            max(0, avg_v - margin_v)
        ], dtype=np.uint8)
        
        self.skin_upper = np.array([
            min(179, avg_h + margin_h),
            min(255, avg_s + margin_s),
            min(255, avg_v + margin_v)
        ], dtype=np.uint8)
        
        self.calibrated = True
        print("✓ Skin color calibrated!")
        print(f"  H: {self.skin_lower[0]}-{self.skin_upper[0]}")
        print(f"  S: {self.skin_lower[1]}-{self.skin_upper[1]}")
        print(f"  V: {self.skin_lower[2]}-{self.skin_upper[2]}")
        print("Ready to draw! Move your hand around and your finger tip will draw.")
    
    def detect_hand(self, frame):
        """
        Detect hand in frame using skin color segmentation with improved reliability.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            tuple: (hand_center, hand_contour) or (None, None) if no hand detected
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply skin color threshold
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Morphological operations - IMPROVED: More aggressive cleaning
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # Additional dilation to connect nearby regions
        mask = cv2.dilate(mask, kernel_large, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Get largest contour (the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area - RELAXED THRESHOLD for better detection
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 1000:  # Reduced from 2000 to be more sensitive
            return None, None
        
        # Calculate center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy), largest_contour
    
    def detect_finger_tip(self, frame, contour, hand_center):
        """
        Detect the index finger tip from hand contour.
        Uses multiple methods for robust detection.
        
        Args:
            frame (np.ndarray): Input video frame
            contour: Hand contour
            hand_center: Center of hand for reference
            
        Returns:
            tuple: (x, y) coordinates of finger tip, or None
        """
        if contour is None or hand_center is None:
            return None
        
        # METHOD 1: Find extreme points using contour
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        topmost = bottommost = leftmost = rightmost = None
        
        for point in contour:
            x, y = point[0]
            if y < y_min:
                y_min = y
                topmost = (x, y)
            if y > y_max:
                y_max = y
                bottommost = (x, y)
            if x < x_min:
                x_min = x
                leftmost = (x, y)
            if x > x_max:
                x_max = x
                rightmost = (x, y)
        
        # METHOD 2: Use convex hull for finger detection
        hull = cv2.convexHull(contour)
        
        # Find multiple candidate points
        candidates = []
        
        # Add topmost point (usually a finger)
        if topmost:
            dist_to_center = np.sqrt((topmost[0] - hand_center[0])**2 + 
                                    (topmost[1] - hand_center[1])**2)
            candidates.append((topmost, dist_to_center, 'top'))
        
        # Add hull extreme points
        for point in hull:
            pt = tuple(point[0])
            dist_to_center = np.sqrt((pt[0] - hand_center[0])**2 + 
                                    (pt[1] - hand_center[1])**2)
            # Prefer points that are away from center (likely fingers)
            if dist_to_center > 30:  # At least 30 pixels from center
                candidates.append((pt, dist_to_center, 'hull'))
        
        # Sort by distance from center - choose furthest point
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            finger_tip = candidates[0][0]
            return finger_tip
        
        # Fallback: return topmost point
        if topmost:
            return topmost
        
        return None
    
    def count_fingers(self, contour):
        """
        Estimate number of extended fingers from contour.
        Improved method for better accuracy.
        
        Args:
            contour: Hand contour
            
        Returns:
            int: Estimated number of extended fingers
        """
        if contour is None:
            return 0
        
        # Simplify contour for processing
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) < 4:
            return 1  # Default to single finger if not enough points
        
        # Try to detect defects (valleys between fingers)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return 1
        
        # Count defects (each defect ~= valley between two fingers)
        # More defects = more extended fingers
        defect_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            # Only count significant defects
            if d > 500:  # Depth threshold
                defect_count += 1
        
        # Estimate finger count: defects + 1 (roughly)
        estimated_fingers = min(5, max(1, defect_count // 2 + 1))
        
        return estimated_fingers
    
    def draw_on_canvas(self, point):
        """
        Draw on the canvas at the given point with smooth trajectory.
        
        Args:
            point (tuple): (x, y) coordinates to draw at
        """
        if point is None or not self.is_drawing:
            return
        
        # Draw smooth line between recent points
        if len(self.drawing_trail) > 0:
            for i in range(len(self.drawing_trail)):
                prev_point = self.drawing_trail[i]
                cv2.line(self.canvas, prev_point, point, self.brush_color, self.brush_thickness)
                # Anti-aliasing for smoother lines
                cv2.circle(self.canvas, point, self.brush_thickness // 2, self.brush_color, -1)
        else:
            cv2.circle(self.canvas, point, self.brush_thickness // 2, self.brush_color, -1)
        
        self.drawing_trail.append(point)
    
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
        Draw the user interface elements on the frame with improved display.
        
        Args:
            frame (np.ndarray): Frame to draw UI elements on
            
        Returns:
            np.ndarray: Frame with UI elements
        """
        # Semi-transparent overlay for UI background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (550, 160), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # FPS Counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calibration status
        status = "CALIBRATED ✓" if self.calibrated else "PRESS SPACEBAR"
        status_color = (0, 255, 0) if self.calibrated else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Current Color Display
        color_name = self.palette_keys[self.current_color_index].upper()
        cv2.putText(frame, f"Color: {color_name}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.brush_color, 2)
        cv2.circle(frame, (300, 100), 10, self.brush_color, -1)
        
        # Brush Thickness Display
        cv2.putText(frame, f"Thickness: {self.brush_thickness}px", (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Drawing status
        draw_status = "DRAWING" if self.is_drawing else "READY"
        draw_color = (0, 255, 0) if self.is_drawing else (100, 100, 100)
        cv2.putText(frame, f"Mode: {draw_status}", (350, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
        
        # Control Instructions (bottom)
        instructions = [
            "INDEX FINGER: Draw | 3+ FINGERS: Erase | C: Clear | S: Save",
            "Z/X: Thickness | Q/W: Color | SPACEBAR: Recalibrate | ESC: Exit"
        ]
        
        y_offset = self.frame_height - 65
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """
        Main application loop - handles camera capture, hand tracking, and rendering.
        """
        if not self.initialize_camera():
            return
        
        self.is_running = True
        print("\n" + "="*60)
        print("      AIR CANVAS - Hand Tracking Drawing Application      ")
        print("="*60)
        print("\nCONTROLS:")
        print("  • SPACEBAR: Calibrate skin color (required on first run)")
        print("  • Index Finger: Draw on canvas")
        print("  • 3+ Fingers: Eraser mode")
        print("  • 'C': Clear entire canvas")
        print("  • 'S': Save drawing to PNG file")
        print("  • 'Z': Decrease brush thickness")
        print("  • 'X': Increase brush thickness")
        print("  • 'Q': Previous color")
        print("  • 'W': Next color")
        print("  • 'ESC': Exit application")
        print("\n⚠️  Please press SPACEBAR to calibrate skin color first!")
        print("="*60 + "\n")
        
        frame_skip = 0
        detection_count = 0
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ ERROR: Failed to read frame from camera")
                    break
                
                # Flip frame for mirror-like interaction
                frame = cv2.flip(frame, 1)
                
                # Draw calibration ROI if not calibrated
                if not self.calibrated:
                    h, w = frame.shape[:2]
                    roi_size = 100
                    roi_x = (w - roi_size) // 2
                    roi_y = (h - roi_size) // 2
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (255, 0, 0), 3)
                    cv2.putText(frame, "Put hand in box, press SPACE", (roi_x-150, roi_y-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Detect hand
                hand_center, hand_contour = self.detect_hand(frame)
                
                if hand_center:
                    self.last_hand_center = hand_center
                    detection_count += 1
                    
                    # Get finger tip position with improved detection
                    finger_tip = self.detect_finger_tip(frame, hand_contour, hand_center)
                    
                    # Count fingers for gesture recognition
                    num_fingers = self.count_fingers(hand_contour)
                    
                    # Gesture logic - IMPROVED: Draw with any single finger extended
                    if num_fingers <= 2:  # 1-2 fingers - Drawing mode (more lenient)
                        self.is_drawing = True
                        if finger_tip:
                            # Validate finger tip is reasonable distance from center
                            dist = np.sqrt((finger_tip[0] - hand_center[0])**2 + 
                                         (finger_tip[1] - hand_center[1])**2)
                            if dist > 10:  # At least 10 pixels away
                                self.draw_on_canvas(finger_tip)
                                cv2.circle(frame, finger_tip, 10, self.brush_color, -1)
                                cv2.circle(frame, finger_tip, 12, self.brush_color, 2)
                    
                    elif num_fingers >= 3:  # Multiple fingers - Eraser mode
                        self.is_drawing = False
                        if hand_center:
                            cv2.circle(self.canvas, hand_center, 25, (255, 255, 255), -1)
                            cv2.circle(frame, hand_center, 25, (0, 0, 255), 2)
                        self.drawing_trail.clear()
                    
                    # Draw hand contour for visual feedback
                    cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)
                    cv2.circle(frame, hand_center, 5, (255, 0, 0), -1)  # Center point
                else:
                    self.is_drawing = False
                    self.drawing_trail.clear()
                
                # Blend canvas with camera feed
                output = cv2.addWeighted(frame, 0.4, self.canvas, 0.6, 0)
                
                # Draw UI elements
                output = self.draw_ui(output)
                
                # Add detection status
                detection_status = f"Hand Detections: {detection_count}"
                cv2.putText(output, detection_status, (self.frame_width - 350, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display the output
                cv2.imshow("Air Canvas - Drawing Application", output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n✓ Exiting Air Canvas...")
                    self.is_running = False
                elif key == 32:  # SPACEBAR - Calibrate
                    print("\nCalibrating skin color...")
                    self.calibrate_skin_color(frame)
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
        
        except KeyboardInterrupt:
            print("\n✓ Application interrupted by user")
        
        except Exception as e:
            print(f"❌ ERROR: An unexpected error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and close windows."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("✓ Cleanup complete. Thank you for using Air Canvas!")


def main():
    """Entry point for the Air Canvas application."""
    app = AirCanvas(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
