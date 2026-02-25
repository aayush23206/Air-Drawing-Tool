# Air Canvas - Hand Tracking Drawing Application

## 📋 Overview

Air Canvas is a professional-grade interactive drawing application that uses real-time hand tracking and gesture recognition to enable users to draw on a virtual canvas using hand gestures. Built with OpenCV and MediaPipe, it provides a seamless experience for intuitive drawing without requiring a physical input device.

## ✨ Features

### Core Functionality
- **Real-time Hand Tracking**: Detects and tracks hand position with MediaPipe pose estimation
- **Drawing Mode**: Use your index finger to draw on the canvas
- **Eraser Mode**: Use 3+ fingers extended to erase portions of the canvas
- **Color Palette**: 8 built-in colors (Green, Blue, Red, Yellow, Purple, Cyan, White, Black)
- **Adjustable Brush Thickness**: 5 preset thickness levels (3px, 5px, 8px, 12px, 15px)

### Performance & UI
- **FPS Counter**: Real-time performance metrics displayed on screen
- **Clean UI Layout**: Intuitive controls and status displays
- **Smooth Drawing**: Trajectory interpolation for continuous, smooth strokes
- **Canvas Blending**: Semi-transparent overlay for visual feedback

### Additional Features
- **Clear Canvas**: Quick gesture to clear the entire drawing
- **Save Drawings**: Export drawings as PNG files with timestamps
- **Error Handling**: Graceful camera detection and error management
- **Modular Architecture**: Clean, extensible code structure

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam/Camera device
- Windows/macOS/Linux

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd "Air Drawing Tool"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import cv2, mediapipe, numpy; print('All dependencies installed!')"
   ```

## 🎮 Usage

### Starting the Application
```bash
python main.py
```

### Controls
| Action | Control | Effect |
|--------|---------|--------|
| **Draw** | Index Finger Extended | Draws on canvas |
| **Erase** | 3+ Fingers Extended | Erases area around finger |
| **Clear Canvas** | 'C' | Clears entire drawing |
| **Save Drawing** | 'S' | Saves drawing to PNG file |
| **Decrease Thickness** | 'Z' | Makes brush thinner |
| **Increase Thickness** | 'X' | Makes brush thicker |
| **Previous Color** | 'Q' | Switches to previous color |
| **Next Color** | 'W' | Switches to next color |
| **Exit Application** | 'ESC' | Closes the application |

## 🏗️ Project Structure

```
Air Drawing Tool/
├── main.py                 # Entry point - starts the application
├── air_canvas.py          # Main AirCanvas class with core functionality
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📚 Code Architecture

### AirCanvas Class
The main application class that handles:
- **Camera initialization and management**
- **Hand detection and landmark extraction**
- **Gesture recognition** (counting extended fingers)
- **Drawing logic** with smooth trajectory
- **UI rendering** and FPS calculation
- **Event handling** for keyboard controls

### Key Methods

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize app with camera settings and parameters |
| `initialize_camera()` | Set up camera capture and canvas |
| `detect_hands()` | Use MediaPipe to find hands in frame |
| `get_finger_position()` | Extract index finger coordinates |
| `count_extended_fingers()` | Gesture recognition logic |
| `draw_on_canvas()` | Render drawing with smooth interpolation |
| `draw_ui()` | Display controls and metrics |
| `run()` | Main application loop |
| `cleanup()` | Release resources |

## 🎨 Customization

### Change Color Palette
Edit the `color_palette` dictionary in `AirCanvas.__init__()`:
```python
self.color_palette = {
    "your_color": (B, G, R),  # OpenCV uses BGR format
    ...
}
```

### Adjust Brush Thickness Options
Edit the `thickness_options` list:
```python
self.thickness_options = [3, 5, 8, 12, 15]
```

### Modify Camera Resolution
Edit in `initialize_camera()`:
```python
self.frame_width = 1280   # Your desired width
self.frame_height = 720   # Your desired height
```

## 🔧 Technical Details

### Dependencies
- **OpenCV (cv2)**: Video capture and image processing
- **MediaPipe**: Hand detection and landmark tracking
- **NumPy**: Array operations and image manipulation

### Hand Landmark Model
Uses MediaPipe's pre-trained hand model with 21 landmarks per hand:
- Landmark 8: Index finger tip (used for drawing)
- Landmark 6-20: Other finger joints (used for gesture recognition)

### Drawing Algorithm
1. Detects index finger position
2. Interpolates smooth line between recent positions
3. Renders circle at current position
4. Blends with semi-transparent canvas overlay

## ⚠️ Troubleshooting

### Camera Not Detected
- Check if camera device is properly connected
- Verify camera is not in use by another application
- Try different `camera_index` values (0, 1, 2, etc.)

### Poor Hand Detection
- Ensure adequate lighting in the room
- Keep hand clearly visible and within frame
- Use neutral background for better contrast
- Adjust MediaPipe detection confidence if needed

### Performance Issues
- Reduce frame resolution in `initialize_camera()`
- Check camera FPS counter - should be 20+ FPS
- Close background applications using camera

## 📊 Performance Metrics

Typical performance on a modern system:
- **Detection Confidence**: 70% (configurable)
- **Tracking Quality**: 50%+ (configurable)
- **Target FPS**: 30 FPS
- **Resolution**: 1280x720

## 🎓 Learning Objectives

This project demonstrates:
- Object-oriented Python programming
- Real-time computer vision with OpenCV
- Machine learning model integration (MediaPipe)
- Gesture recognition algorithms
- Event-driven programming
- Resource management and cleanup

## 📝 License

This project is created for educational purposes.

## 🤝 Contributing

Feel free to extend this project with:
- Additional gestures (two-finger drawing, rotation-based rotation, etc.)
- Shape drawing (lines, rectangles, circles)
- Undo/Redo functionality
- Multiple canvas layers
- Screenshot functionality
- Touch screen support

## 👨‍💻 Author

Created as a professional-grade demo of hand tracking technology for portfolio purposes.

---

**Enjoy creating with Air Canvas! 🎨**
