import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Face Mesh solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define lipstick shades with BGR format (OpenCV uses BGR instead of RGB)
LIPSTICK_COLORS = [
    (43, 32, 212),  # Red
    (164, 73, 163),  # Pink
    (0, 0, 128),    # Maroon
    (172, 79, 57),  # Royal Blue
    (133, 58, 133), # Purple
    (75, 177, 243)  # Coral
]

# Full set of lip landmarks from MediaPipe Face Mesh
# Complete lips outline (upper and lower)
LIPS_ALL = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 
           61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
           78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Inner lips contour
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Specific lip landmarks for better accuracy
UPPER_LIP_CENTER = [0, 267, 13, 14, 17]
LOWER_LIP_CENTER = [17, 14, 13, 84, 181]

def create_convex_hull(landmarks, indices, image_shape):
    """Create a convex hull from landmarks for better coverage"""
    h, w = image_shape[:2]
    points = []
    
    for idx in indices:
        pt = landmarks.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        points.append((x, y))
    
    # Convert points to numpy array for hull calculation
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    return hull

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
    highlight_intensity = 0.3
    
    # Apply the highlight effect using proper scalar values
    highlight_layer = np.ones_like(image) * 255
    highlight_mask_3d = np.stack([highlight_mask, highlight_mask, highlight_mask], axis=2) / 255.0
    
    # Use element-wise multiplication for the mask instead of max operation
    for c in range(3):  # Apply to each channel
        result[:,:,c] = result[:,:,c] * (1 - highlight_intensity * highlight_mask_3d[:,:,c]) + \
                        highlight_layer[:,:,c] * (highlight_intensity * highlight_mask_3d[:,:,c])
    
    return result

def draw_color_options(image, current_idx):
    """Draw the color options UI on the image"""
    h, w, _ = image.shape
    text = f"Lipstick Color: {current_idx+1}/{len(LIPSTICK_COLORS)}"
    
    # Draw text with background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_bg_x1, text_bg_y1 = 20, 70
    text_bg_x2, text_bg_y2 = text_bg_x1 + text_size[0] + 20, text_bg_y1 + text_size[1] + 20
    
    cv2.rectangle(image, (text_bg_x1-5, text_bg_y1-5), (text_bg_x2+5, text_bg_y2+5), (0, 0, 0), -1)
    cv2.putText(image, text, (text_bg_x1, text_bg_y1 + text_size[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw color swatches
    swatch_size = 30
    swatch_space = 10
    swatch_y = text_bg_y2 + 20
    
    for i, color in enumerate(LIPSTICK_COLORS):
        swatch_x = 20 + i * (swatch_size + swatch_space)
        if i == current_idx:
            # Draw highlight for selected color
            cv2.rectangle(image, (swatch_x-3, swatch_y-3), 
                         (swatch_x+swatch_size+3, swatch_y+swatch_size+3), 
                         (255, 255, 255), 2)
        
        cv2.rectangle(image, (swatch_x, swatch_y), 
                     (swatch_x+swatch_size, swatch_y+swatch_size), 
                     color, -1)
    
    # Add usage instructions
    instructions = "Press 'n' for next color, 'p' for previous, 'h' to hide/show mesh"
    cv2.putText(image, instructions, (20, swatch_y + swatch_size + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Current lipstick color index
    color_idx = 0
    
    # Flag to show/hide mesh
    show_mesh = True
    
    # Setup the face mesh instance
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        # Initialize previous landmarks for stability
        prev_landmarks = None
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # To improve performance, optionally mark the image as not writeable
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            
            # Draw the face landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Create a clean copy for lipstick application
            clean_image = image.copy()
            
            if results.multi_face_landmarks:
                # Use current landmarks
                face_landmarks = results.multi_face_landmarks[0]
                prev_landmarks = face_landmarks  # Store landmarks for next frame
                
                # Apply lipstick on clean image
                image = apply_lipstick(clean_image, face_landmarks, 
                                      LIPSTICK_COLORS[color_idx])
                
                if show_mesh:
                    # Draw the face landmarks
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw the face contours
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Draw the eye landmarks
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
            elif prev_landmarks:
                # If no landmarks detected in current frame but we have previous ones
                # Use previous landmarks for stability
                image = apply_lipstick(clean_image, prev_landmarks, 
                                      LIPSTICK_COLORS[color_idx])
            
            # Draw UI with color options
            draw_color_options(image, color_idx)
            
            # Display face landmark detection title
            cv2.putText(image, f'Face Landmark Detection with Lipstick', (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow('MediaPipe Face Landmark Detection', image)
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('n'):  # Next color
                color_idx = (color_idx + 1) % len(LIPSTICK_COLORS)
            elif key == ord('p'):  # Previous color
                color_idx = (color_idx - 1) % len(LIPSTICK_COLORS)
            elif key == ord('h'):  # Toggle mesh visibility
                show_mesh = not show_mesh
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()