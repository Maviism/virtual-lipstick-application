"""
Utility functions for the Virtual Makeup Application
"""
import cv2
import numpy as np

def create_convex_hull(landmarks, landmark_indices, image_shape):
    """
    Create a convex hull from facial landmarks
    
    Parameters:
    - landmarks: MediaPipe face landmarks
    - landmark_indices: List of landmark indices to use
    - image_shape: Shape of the image (height, width, channels)
    
    Returns:
    - np.array: Convex hull points as numpy array
    """
    h, w = image_shape[:2]
    points = []
    
    for idx in landmark_indices:
        pt = landmarks.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        points.append((x, y))
    
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    return hull

def remove_background(frame, segmentation_result, bg_type, bg_color, bg_image, default_bg_image=None):
    """
    Remove background from an image frame based on segmentation result
    
    Parameters:
    - frame: The input frame (BGR format)
    - segmentation_result: MediaPipe selfie segmentation result
    - bg_type: Type of background ("color" or "image")
    - bg_color: Background color in BGR format
    - bg_image: Background image (if bg_type is "image")
    - default_bg_image: Default background image to use if bg_image is None
    
    Returns:
    - np.array: Frame with background replaced
    """
    # Create a mask of the person
    condition = np.stack(
        (segmentation_result.segmentation_mask,) * 3, axis=-1
    ) > 0.1
    
    # Process background based on selected option
    if bg_type == "color":
        # Create solid color background
        bg = np.ones(frame.shape, dtype=np.uint8)
        bg[:] = bg_color
        
    elif bg_type == "image" and bg_image is not None:
        # Resize background image to match frame size
        bg = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    else:
        # Use default background if available, else green background
        if default_bg_image is not None:
            bg = cv2.resize(default_bg_image, (frame.shape[1], frame.shape[0]))
        else:
            # Default to green background if image is not available
            bg = np.ones(frame.shape, dtype=np.uint8)
            bg[:] = (0, 128, 0)  # Green default
    
    # Apply the foreground mask and background image
    return np.where(condition, frame, bg)

