# Virtual Makeup Application

A computer vision application that applies virtual lipstick in real-time using webcam or static images.

![Virtual Makeup App](https://github.com/yourusername/virtual-makeup-app/raw/main/screenshot.jpg)

## Features

- Real-time lipstick application using webcam
- Support for uploading and processing static images
- Multiple preset lipstick colors (Red, Pink, Coral, Nude, Burgundy, Purple, Orange)
- Custom color picker for personalized lipstick shades
- Background removal with options for solid color or custom image background
- Face mesh visualization option for debugging
- Camera switching support for systems with multiple cameras
- User-friendly GUI interface built with tkinter

## Requirements

- Python 3.12.10
- OpenCV
- MediaPipe
- NumPy
- tkinter
- PIL (Pillow)

## Installation

1. Clone this repository

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Use the GUI to:
   - Select preset lipstick colors
   - Choose custom lipstick colors
   - Toggle background removal
   - Select background type (solid color or image)
   - Switch between webcam and static image modes
   - Show/hide face mesh for debugging

### Keyboard Shortcuts
- Press `k` to switch between available cameras

## Project Structure

```
virtual-makeup-app/
│
├── main.py              # Main application entry point
├── requirements.txt     # Package dependencies
├── default-bg.jpg       # Default background image
│
└── modules/             # Application modules
    ├── __init__.py
    ├── constants.py     # Landmark indices and color presets
    ├── lipstick_processor.py  # Lipstick application algorithm
    └── utils.py         # Utility functions
```

## How It Works

The application uses MediaPipe's Face Mesh to detect facial landmarks, particularly focusing on the lips. It then creates masks for the lip area and applies color with natural blending using an overlay technique. For background removal, it uses MediaPipe's Selfie Segmentation to separate the subject from the background.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for providing the face detection and mesh models
- [OpenCV](https://opencv.org/) for image processing capabilities