"""
Air Canvas - Main Entry Point

A professional hand-tracking based drawing application.
Run this script to start the Air Canvas application.
"""

from air_canvas import AirCanvas


def main():
    """Start the Air Canvas application."""
    app = AirCanvas(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
