"""
Lipstick processing logic for the Virtual Makeup Application
"""
import cv2
import numpy as np
from .utils import create_convex_hull
from .constants import LIPS_ALL, INNER_LIPS

def apply_lipstick(image, landmarks, color, intensity=0.8, blur_amount=5):
    """
    Apply lipstick shade to lips using face landmarks with improved technique
    
    Parameters:
    - image: Input image
    - landmarks: Face landmarks from MediaPipe
    - color: BGR color for lipstick
    - intensity: Opacity/intensity of lipstick (0.0 to 1.0)
    - blur_amount: Amount of blur to apply for smoother edges
    
    Returns:
    - Image with lipstick applied
    """
    # Create a separate layer for the lipstick
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create a convex hull for the entire lips for better coverage
    lips_hull = create_convex_hull(landmarks, LIPS_ALL, image.shape)
    
    # Fill the outer lips area
    cv2.fillPoly(mask, [lips_hull], 255)
    
    # Get inner lip points
    inner_points = []
    for idx in INNER_LIPS:
        pt = landmarks.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        inner_points.append((x, y))
    
    # Create inner lips convex hull
    inner_points = np.array(inner_points, dtype=np.int32)
    inner_hull = cv2.convexHull(inner_points)
    
    # Cut out inner lips
    cv2.fillPoly(mask, [inner_hull], 0)
    
    # Apply slight dilation to ensure coverage
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Apply blur for smoother edges
    mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
    
    # Create a color adjustment layer for more natural look
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = np.zeros_like(hsv)
    
    # Extract BGR color components
    b, g, r = color
    
    # Convert to HSV for better blending
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
    h_value, s_value, _ = color_hsv
    
    # Set HSV values for the lipstick
    hsv_mask[:, :, 0] = h_value  # Hue from lipstick color
    hsv_mask[:, :, 1] = s_value  # Saturation from lipstick color
    hsv_mask[:, :, 2] = hsv[:, :, 2]  # Keep original brightness
    
    # Convert back to BGR
    color_layer = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    
    # Apply the mask
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    
    # Use overlay blending mode for more natural look
    def overlay_blend(bg, fg, alpha):
        # Apply overlay blend mode formula
        fg_norm = fg / 255.0
        bg_norm = bg / 255.0
        
        # Overlay blend mode formula
        blend = np.zeros_like(bg_norm)
        # Where background <= 0.5
        mask_dark = bg_norm <= 0.5
        blend[mask_dark] = 2 * bg_norm[mask_dark] * fg_norm[mask_dark]
        # Where background > 0.5
        mask_light = ~mask_dark
        blend[mask_light] = 1 - 2 * (1 - bg_norm[mask_light]) * (1 - fg_norm[mask_light])
        
        # Apply alpha blending
        result = alpha * blend + (1 - alpha) * bg_norm
        return (result * 255).astype(np.uint8)
    
    # Apply overlay blend
    b_channel = overlay_blend(image[:,:,0], color_layer[:,:,0], intensity * mask_3d[:,:,0])
    g_channel = overlay_blend(image[:,:,1], color_layer[:,:,1], intensity * mask_3d[:,:,1])
    r_channel = overlay_blend(image[:,:,2], color_layer[:,:,2], intensity * mask_3d[:,:,2])
    
    # Combine channels
    result = cv2.merge([b_channel, g_channel, r_channel])
    
    # Add a subtle highlight for a glossy effect
    highlight_mask = mask.copy()
    highlight_mask = cv2.dilate(highlight_mask, np.ones((2, 2), np.uint8), iterations=1)
    highlight_mask = cv2.erode(highlight_mask, np.ones((5, 5), np.uint8), iterations=1)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
    
    # Create a fixed scalar value for highlight intensity (0.3)
    highlight_intensity = 0.05
    
    # Apply the highlight effect using proper scalar values
    highlight_layer = np.ones_like(image) * 255
    highlight_mask_3d = np.stack([highlight_mask, highlight_mask, highlight_mask], axis=2) / 255.0
    
    # Use element-wise multiplication for the mask instead of max operation
    for c in range(3):  # Apply to each channel
        result[:,:,c] = result[:,:,c] * (1 - highlight_intensity * highlight_mask_3d[:,:,c]) + \
                        highlight_layer[:,:,c] * (highlight_intensity * highlight_mask_3d[:,:,c])
    
    return result