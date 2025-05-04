import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
from PIL import Image, ImageTk
import os

# Initialize MediaPipe Face Detection and Face Mesh solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation  # Add selfie segmentation

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

# New tkinter GUI wrapper class
class VirtualMakeUpApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Create a frame for the webcam feed
        self.frame_webcam = ttk.Frame(window)
        self.frame_webcam.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Create a label for the webcam feed
        self.label_webcam = ttk.Label(self.frame_webcam)
        self.label_webcam.pack()
        
        # Create a frame for the controls
        self.frame_controls = ttk.Frame(window)
        self.frame_controls.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        # Create a frame for preset colors
        self.frame_presets = ttk.LabelFrame(self.frame_controls, text="Preset Colors")
        self.frame_presets.pack(fill=tk.X, padx=5, pady=5)
        
        # Add color buttons for presets with color as button background
        self.preset_buttons_frame = ttk.Frame(self.frame_presets)
        self.preset_buttons_frame.pack(fill=tk.X, padx=2, pady=5)
        
        for i, color in enumerate(LIPSTICK_COLORS):
            # Convert BGR to RGB for tkinter
            rgb_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
            
            # Calculate brightness to determine if text should be black or white
            brightness = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]) / 255
            text_color = "black" if brightness > 0.5 else "white"
            
            # Create a button with the color as background
            btn = tk.Button(
          self.preset_buttons_frame, 
          text=f"Color {i+1}",
          bg=rgb_color,
          fg=text_color,
          width=10,
          command=lambda c=color, i=i: self.select_preset_color(c, i)
            )
            
            btn.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Add a custom color picker button
        ttk.Button(self.frame_controls, text="Choose Custom Color", 
                   command=self.choose_custom_color).pack(fill=tk.X, padx=5, pady=5)
        
        # Display current color
        self.frame_current_color = ttk.LabelFrame(self.frame_controls, text="Current Color")
        self.frame_current_color.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_color_canvas = tk.Canvas(self.frame_current_color, width=50, height=30, bg='red')
        self.current_color_canvas.pack(padx=5, pady=5)
        
        # Background removal checkbox
        self.remove_bg_var = tk.BooleanVar(value=True)
        self.remove_bg_checkbox = ttk.Checkbutton(self.frame_controls, text="Remove Background", 
                                                variable=self.remove_bg_var)
        self.remove_bg_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # Background options frame
        self.frame_bg_options = ttk.LabelFrame(self.frame_controls, text="Background Options")
        self.frame_bg_options.pack(fill=tk.X, padx=5, pady=5)
        
        # Background type radio buttons
        self.bg_type = tk.StringVar(value="image")
        
        ttk.Radiobutton(self.frame_bg_options, text="Solid Color", 
                       variable=self.bg_type, value="color").pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(self.frame_bg_options, text="Image Background", 
                       variable=self.bg_type, value="image").pack(anchor=tk.W, padx=5, pady=2)
        
        # Background color picker
        self.bg_color_frame = ttk.Frame(self.frame_bg_options)
        self.bg_color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.bg_color_frame, text="Background Color:").pack(side=tk.LEFT, padx=2)
        
        self.bg_color = (0, 128, 0)  # Default green background (in BGR)
        
        self.bg_color_canvas = tk.Canvas(self.bg_color_frame, width=30, height=20, bg='#008000')
        self.bg_color_canvas.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.bg_color_frame, text="Choose", width=8, 
                  command=self.choose_bg_color).pack(side=tk.LEFT, padx=2)
        
        # Background image selection
        ttk.Button(self.frame_bg_options, text="Choose Background Image", 
                  command=self.choose_bg_image).pack(fill=tk.X, padx=5, pady=5)
        
        self.bg_image_label = ttk.Label(self.frame_bg_options, text="No image selected")
        self.bg_image_label.pack(padx=5, pady=2)
        
        # Store the background image
        self.bg_image = None
        self.bg_image_path = None
        
        # Load default background image
        default_bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default-bg.jpg")
        if os.path.exists(default_bg_path):
            self.default_bg_image = cv2.imread(default_bg_path)
        else:
            self.default_bg_image = None
        
        # Checkbox for showing face mesh
        self.show_mesh_var = tk.BooleanVar(value=False)
        self.show_mesh_checkbox = ttk.Checkbutton(self.frame_controls, text="Show Face Mesh", 
                                                variable=self.show_mesh_var)
        self.show_mesh_checkbox.pack(anchor=tk.W, padx=5, pady=10)
        
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
        
        # Initialize MediaPipe Selfie Segmentation
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0  # 0 for general model, 1 for landscape model
        )
        
        # Previous landmarks for stability
        self.prev_landmarks = None
        
        # Start the video capture loop
        self.update()
        
        # Set window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def choose_bg_color(self):
        """Open color picker for background color"""
        # Convert BGR to RGB for the color picker
        current_rgb = (self.bg_color[2], self.bg_color[1], self.bg_color[0])
        
        color_result = colorchooser.askcolor(initialcolor=current_rgb, title="Choose Background Color")
        
        if color_result[0]:  # If user didn't cancel
            rgb_color = color_result[0]
            # Convert RGB to BGR for OpenCV
            self.bg_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            
            # Update the color display
            self.bg_color_canvas.config(bg=color_result[1])
            
            # Select the color option
            self.bg_type.set("color")
    
    def choose_bg_image(self):
        """Open file dialog to select a background image"""
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load the image
                self.bg_image_path = file_path
                self.bg_image = cv2.imread(file_path)
                
                # Update the display with the filename
                filename = os.path.basename(file_path)
                if len(filename) > 25:
                    filename = filename[:22] + "..."
                self.bg_image_label.config(text=f"Selected: {filename}")
                
                # Select the image option
                self.bg_type.set("image")
            except Exception as e:
                print(f"Error loading image: {str(e)}")
    
    def select_preset_color(self, color, idx):
        """Set lipstick color to a preset color"""
        self.current_color = color
        self.color_idx = idx
        self.update_color_display()
        
    def choose_custom_color(self):
        """Open color picker for custom lipstick color"""
        # Convert BGR to RGB for the color picker
        current_rgb = (self.current_color[2], self.current_color[1], self.current_color[0])
        color_result = colorchooser.askcolor(initialcolor=current_rgb, title="Choose Lipstick Color")
        
        if color_result[0]:  # If user didn't cancel
            rgb_color = color_result[0]
            # Convert RGB to BGR for OpenCV
            self.current_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            self.color_idx = -1  # Custom color
            self.update_color_display()
    
    def update_color_display(self):
        """Update the current color display"""
        # Convert BGR to RGB for tkinter
        rgb_color = f'#{self.current_color[2]:02x}{self.current_color[1]:02x}{self.current_color[0]:02x}'
        self.current_color_canvas.config(bg=rgb_color)
    
    def update(self):
        """Update the video frame"""
        ret, frame = self.webcam.read()
        
        if ret:
            # Process the frame
            frame = cv2.flip(frame, 1)  # Mirror the image for a more intuitive view
            
            # Create a clean copy for lipstick application
            clean_frame = frame.copy()
            
            # Process with MediaPipe Face Mesh
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(frame_rgb)
            
            # Background removal if enabled
            if self.remove_bg_var.get():
                # Process with MediaPipe Selfie Segmentation
                segmentation_results = self.selfie_segmentation.process(frame_rgb)
                
                # Create a mask of the person
                condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
                
                # Process background based on selected option
                if self.bg_type.get() == "color":
                    # Create solid color background
                    bg_image = np.ones(frame.shape, dtype=np.uint8)
                    bg_image[:] = self.bg_color
                    
                elif self.bg_type.get() == "image" and self.bg_image is not None:
                    # Resize background image to match frame size
                    bg_image = cv2.resize(self.bg_image, (frame.shape[1], frame.shape[0]))
                else:
                    # Use default background if available, else green background
                    if self.default_bg_image is not None:
                        bg_image = cv2.resize(self.default_bg_image, (frame.shape[1], frame.shape[0]))
                    else:
                        # Default to green background if image is not available
                        bg_image = np.ones(frame.shape, dtype=np.uint8)
                        bg_image[:] = (0, 128, 0)  # Green default
                
                # Apply the foreground mask and background image
                frame = np.where(condition, frame, bg_image)
                clean_frame = frame.copy()  # Update clean frame with background removed
            
            frame.flags.writeable = True
                
            # Set lipstick intensity and blur amount
            intensity = 0.8
            blur_amount = 5
            show_mesh = self.show_mesh_var.get()
            
            # If even blur amount, make it odd (required by GaussianBlur)
            if blur_amount % 2 == 0:
                blur_amount += 1
                
            if face_results.multi_face_landmarks:
                # Use current landmarks
                face_landmarks = face_results.multi_face_landmarks[0]
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
        self.selfie_segmentation.close()
        self.window.destroy()

def main():
    # Create the root window
    root = tk.Tk()
    root.title("Makeup Virtual Try-On")
    root.resizable(False, False)  # Disable resizing
    
    # Set a more modern theme if available
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')  # Using 'clam' theme for better looking controls
    except:
        pass  # If theming fails, use default
    
    # Create the app
    app = VirtualMakeUpApp(root, "MakeUp Virtual Try-On")
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()