import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk

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
    highlight_intensity = 0.05
    
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

# Original main function - kept for reference but will be replaced by the tkinter version
def main_original():
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

# New tkinter GUI wrapper class
class LipstickApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Create a frame for the webcam feed
        self.frame_webcam = ttk.Frame(window)
        self.frame_webcam.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Create a label for the webcam feed
        self.label_webcam = ttk.Label(self.frame_webcam)
        self.label_webcam.pack()
        
        # Create a frame for the controls
        self.frame_controls = ttk.Frame(window)
        self.frame_controls.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        # Add a section title for lipstick settings
        ttk.Label(self.frame_controls, text="Lipstick Settings", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create a frame for preset colors
        self.frame_presets = ttk.LabelFrame(self.frame_controls, text="Preset Colors")
        self.frame_presets.pack(fill=tk.X, padx=5, pady=5)
        
        # Add color buttons for presets
        self.preset_buttons = []
        for i, color in enumerate(LIPSTICK_COLORS):
            # Convert BGR to RGB for tkinter
            rgb_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
            btn = ttk.Button(self.frame_presets, text=f"Color {i+1}", 
                            command=lambda c=color, i=i: self.select_preset_color(c, i))
            btn.pack(side=tk.LEFT, padx=2, pady=5)
            self.preset_buttons.append(btn)
            
            # Color indicator next to button
            color_indicator = tk.Canvas(self.frame_presets, width=15, height=15, bg=rgb_color)
            color_indicator.pack(side=tk.LEFT, padx=(0, 5), pady=5)
        
        # Add a custom color picker button
        ttk.Button(self.frame_controls, text="Choose Custom Color", 
                   command=self.choose_custom_color).pack(fill=tk.X, padx=5, pady=5)
        
        # Display current color
        self.frame_current_color = ttk.LabelFrame(self.frame_controls, text="Current Color")
        self.frame_current_color.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_color_canvas = tk.Canvas(self.frame_current_color, width=50, height=30, bg='red')
        self.current_color_canvas.pack(padx=5, pady=5)
        
        # Add sliders for parameters
        self.frame_sliders = ttk.LabelFrame(self.frame_controls, text="Effect Parameters")
        self.frame_sliders.pack(fill=tk.X, padx=5, pady=5)
        
        # Intensity slider
        ttk.Label(self.frame_sliders, text="Color Intensity:").pack(anchor=tk.W, padx=5)
        self.intensity_var = tk.DoubleVar(value=0.8)
        self.intensity_slider = ttk.Scale(self.frame_sliders, from_=0.1, to=1.0, 
                                          variable=self.intensity_var, orient=tk.HORIZONTAL, length=200)
        self.intensity_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # Blur amount slider
        ttk.Label(self.frame_sliders, text="Edge Smoothness:").pack(anchor=tk.W, padx=5)
        self.blur_var = tk.IntVar(value=5)
        self.blur_slider = ttk.Scale(self.frame_sliders, from_=1, to=15, 
                                     variable=self.blur_var, orient=tk.HORIZONTAL, length=200)
        self.blur_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # Highlight intensity slider
        ttk.Label(self.frame_sliders, text="Glossy Effect:").pack(anchor=tk.W, padx=5)
        self.highlight_var = tk.DoubleVar(value=0.05)
        self.highlight_slider = ttk.Scale(self.frame_sliders, from_=0.0, to=0.3, 
                                          variable=self.highlight_var, orient=tk.HORIZONTAL, length=200)
        self.highlight_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # Checkbox for showing face mesh
        self.show_mesh_var = tk.BooleanVar(value=True)
        self.show_mesh_checkbox = ttk.Checkbutton(self.frame_controls, text="Show Face Mesh", 
                                                variable=self.show_mesh_var)
        self.show_mesh_checkbox.pack(anchor=tk.W, padx=5, pady=10)
        
        # Add a screenshot button
        ttk.Button(self.frame_controls, text="Take Screenshot", 
                  command=self.take_screenshot).pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.statusbar = ttk.Label(window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize variables
        self.webcam = cv2.VideoCapture(0)
        self.color_idx = 0
        self.current_color = LIPSTICK_COLORS[self.color_idx]
        self.update_color_display()
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Previous landmarks for stability
        self.prev_landmarks = None
        
        # Start the video capture loop
        self.update()
        
        # Set window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def select_preset_color(self, color, idx):
        """Set lipstick color to a preset color"""
        self.current_color = color
        self.color_idx = idx
        self.update_color_display()
        self.status_var.set(f"Selected preset color {idx+1}")
        
    def choose_custom_color(self):
        """Open color picker for custom lipstick color"""
        # Convert BGR to RGB for the color picker
        current_rgb = (self.current_color[2], self.current_color[1], self.current_color[0])
        color_result = colorchooser.askcolor(rgb=current_rgb, title="Choose Lipstick Color")
        
        if color_result[0]:  # If user didn't cancel
            rgb_color = color_result[0]
            # Convert RGB to BGR for OpenCV
            self.current_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            self.color_idx = -1  # Custom color
            self.update_color_display()
            self.status_var.set("Custom color selected")
    
    def update_color_display(self):
        """Update the current color display"""
        # Convert BGR to RGB for tkinter
        rgb_color = f'#{self.current_color[2]:02x}{self.current_color[1]:02x}{self.current_color[0]:02x}'
        self.current_color_canvas.config(bg=rgb_color)
    
    def take_screenshot(self):
        """Save current frame as screenshot"""
        if hasattr(self, 'current_frame'):
            timestamp = cv2.getTickCount()
            filename = f"lipstick_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.status_var.set(f"Screenshot saved as {filename}")
    
    def update(self):
        """Update the video frame"""
        ret, frame = self.webcam.read()
        
        if ret:
            # Process the frame
            frame = cv2.flip(frame, 1)  # Mirror the image for a more intuitive view
            
            # Process with MediaPipe
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            frame.flags.writeable = True
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Create a clean copy for lipstick application
            clean_frame = frame.copy()
            
            # Get current parameters from sliders
            intensity = self.intensity_var.get()
            blur_amount = self.blur_var.get()
            show_mesh = self.show_mesh_var.get()
            
            # If even blur amount, make it odd (required by GaussianBlur)
            if blur_amount % 2 == 0:
                blur_amount += 1
                
            if results.multi_face_landmarks:
                # Use current landmarks
                face_landmarks = results.multi_face_landmarks[0]
                self.prev_landmarks = face_landmarks
                
                # Apply lipstick
                frame = apply_lipstick(clean_frame, face_landmarks, self.current_color, intensity, blur_amount)
                
                if show_mesh:
                    # Draw the face landmarks
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw the face contours
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Draw the eye landmarks
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
            elif self.prev_landmarks:
                # If no landmarks detected in current frame but we have previous ones
                frame = apply_lipstick(clean_frame, self.prev_landmarks, self.current_color, intensity, blur_amount)
            
            # Store the current frame for screenshots
            self.current_frame = frame.copy()
            
            # Convert to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL format and then to ImageTk
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.LANCZOS)  # Resize for display
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the label with new image
            self.label_webcam.imgtk = imgtk
            self.label_webcam.configure(image=imgtk)
        
        # Schedule the next update
        self.window.after(10, self.update)
    
    def on_closing(self):
        """Clean up resources when window is closed"""
        if self.webcam.isOpened():
            self.webcam.release()
        self.face_mesh.close()
        self.window.destroy()

def main():
    # Create the root window
    root = tk.Tk()
    root.title("Lipstick Virtual Try-On")
    
    # Set a more modern theme if available
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')  # Using 'clam' theme for better looking controls
    except:
        pass  # If theming fails, use default
    
    # Create the app
    app = LipstickApp(root, "Lipstick Virtual Try-On GUI")
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()