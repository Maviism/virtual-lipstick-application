import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
from PIL import Image, ImageTk
import os
from modules.lipstick_processor import apply_lipstick 
from modules.constants import LIPSTICK_COLORS 
from modules.utils import remove_background

# Initialize MediaPipe Face Detection and Face Mesh solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation  

# New tkinter GUI wrapper class
class VirtualMakeUpApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
          # Create main container frames
        self.main_container = ttk.Frame(window)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create a container for webcam and input mode
        self.right_container = ttk.Frame(self.main_container)
        self.right_container.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        # Create a frame for the webcam feed
        self.frame_webcam = ttk.Frame(self.right_container)
        self.frame_webcam.pack(side=tk.TOP, padx=0, pady=0)
        
        # Create a label for the webcam feed
        self.label_webcam = ttk.Label(self.frame_webcam)
        self.label_webcam.pack()
        
        # Create a frame for the controls
        self.frame_controls = ttk.Frame(self.main_container)
        self.frame_controls.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        # Create a frame for preset colors
        self.frame_presets = ttk.LabelFrame(self.frame_controls, text="Preset Colors")
        self.frame_presets.pack(fill=tk.X, padx=5, pady=5)
        
        # Add color buttons for presets with color as button background
        self.preset_buttons_frame = ttk.Frame(self.frame_presets)
        self.preset_buttons_frame.pack(fill=tk.X, padx=2, pady=5)
        
        for name, color in LIPSTICK_COLORS.items():
            # Convert BGR to RGB for tkinter
            rgb_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
            
            # Calculate brightness to determine if text should be black or white
            brightness = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]) / 255
            text_color = "black" if brightness > 0.5 else "white"
            
            # Create a button with the color as background
            btn = tk.Button(
                self.preset_buttons_frame, 
                text=name,
                bg=rgb_color,
                fg=text_color,
                width=10,
                command=lambda c=color, n=name: self.select_preset_color(c, n)
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
          # Add image mode section below the webcam frame
        self.frame_image_mode = ttk.LabelFrame(self.right_container, text="Input Mode")
        self.frame_image_mode.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Create a horizontal layout for input mode controls
        self.image_mode_controls = ttk.Frame(self.frame_image_mode)
        self.image_mode_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Left side controls
        self.image_left_frame = ttk.Frame(self.image_mode_controls)
        self.image_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Use image checkbox
        self.use_image_var = tk.BooleanVar(value=False)
        self.use_image_checkbox = ttk.Checkbutton(
            self.image_left_frame, 
            text="Use Image", 
            variable=self.use_image_var,
            command=self.toggle_image_mode
        )
        self.use_image_checkbox.pack(anchor=tk.W, pady=2)
        
        # Label to show selected image name
        self.image_label = ttk.Label(self.image_left_frame, text="No image selected")
        self.image_label.pack(anchor=tk.W, pady=2)
        
        # Right side buttons
        self.image_right_frame = ttk.Frame(self.image_mode_controls)
        self.image_right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Button to select input image
        self.select_image_button = ttk.Button(
            self.image_right_frame, 
            text="Select Image", 
            command=self.choose_input_image
        )
        self.select_image_button.pack(fill=tk.X, pady=2)
        
        # Button to return to webcam mode
        self.return_to_webcam_button = ttk.Button(
            self.image_right_frame, 
            text="Return to Webcam", 
            command=self.use_webcam_mode
        )
        self.return_to_webcam_button.pack(fill=tk.X, pady=2)
        
        # Initialize variables
        self.webcam = cv2.VideoCapture(1)
        self.camera_index = 1  # Track the current camera index
        
        # Get list of color names and set initial color
        self.color_names = list(LIPSTICK_COLORS.keys())
        self.current_color_name = self.color_names[0]
        self.current_color = LIPSTICK_COLORS[self.current_color_name]
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
        
        # Add image upload mode variables
        self.use_image = False
        self.image_path = None
        self.image = None
        
        # Start the video capture loop
        self.update()
        
        # Set window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind key events
        self.window.bind('<k>', self.switch_camera)
        
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
    
    def select_preset_color(self, color, color_name):
        """Set lipstick color to a preset color"""
        self.current_color = color
        self.current_color_name = color_name
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
            self.current_color_name = "Custom"  # Custom color
            self.update_color_display()
    
    def update_color_display(self):
        """Update the current color display"""
        # Convert BGR to RGB for tkinter
        rgb_color = f'#{self.current_color[2]:02x}{self.current_color[1]:02x}{self.current_color[0]:02x}'
        self.current_color_canvas.config(bg=rgb_color)
    
    def update(self):
        """Update the video frame"""
        if self.use_image and self.image is not None:
            # Use the uploaded image
            frame = self.image.copy()
            ret = True
        else:
            # Use webcam
            ret, frame = self.webcam.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror the image for a more intuitive view
        
        if ret:
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
                
                # Use the background removal function from utils module
                frame = remove_background(
                    frame, 
                    segmentation_results, 
                    self.bg_type.get(), 
                    self.bg_color, 
                    self.bg_image, 
                    self.default_bg_image
                )
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
              # Calculate aspect ratio for resizing
            h, w = frame_rgb.shape[:2]
            target_width = 640
            target_height = 480
            
            # Preserve aspect ratio
            if h > 0 and w > 0:
                ratio = min(target_width / w, target_height / h)
                new_size = (int(w * ratio), int(h * ratio))
                
                # Convert to PIL format and then to ImageTk
                img = Image.fromarray(frame_rgb)
                img = img.resize(new_size, Image.LANCZOS)
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

    def choose_input_image(self):
        """Open file dialog to select an input image for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load the image
                self.image_path = file_path
                self.image = cv2.imread(file_path)
                
                # Update the display with the filename
                filename = os.path.basename(file_path)
                if len(filename) > 25:
                    filename = filename[:22] + "..."
                self.image_label.config(text=f"Selected: {filename}")
                
                # Set to image mode
                self.use_image = True
                self.use_image_var.set(True)
                
                print(f"Using image: {filename}")
                
            except Exception as e:
                print(f"Error loading image: {str(e)}")
    
    def toggle_image_mode(self):
        """Toggle between image mode and webcam mode"""
        self.use_image = self.use_image_var.get()
        
        if self.use_image and self.image is None:
            # Prompt to select an image if none is selected
            self.choose_input_image()
            
    def use_webcam_mode(self):
        """Switch back to webcam mode"""
        self.use_image = False
        self.use_image_var.set(False)
        print("Returned to webcam mode")
        
    def switch_camera(self, event=None):
        """Switch between available cameras"""
        # Close current camera
        if self.webcam.isOpened():
            self.webcam.release()
        
        # Try to open next camera
        next_index = (self.camera_index + 1) % 4  # Cycle through cameras 0, 1, 2
        self.webcam = cv2.VideoCapture(next_index)
        
        # If the camera opened successfully, update the index
        if self.webcam.isOpened():
            self.camera_index = next_index
            print(f"Switched to camera {self.camera_index}")
        else:
            # If failed, try camera 0 as fallback
            self.webcam = cv2.VideoCapture(0)
            if self.webcam.isOpened():
                self.camera_index = 0
                print(f"Switched to camera {self.camera_index}")
            else:
                print("Failed to switch camera")
                # Try to reopen the previous camera
                self.webcam = cv2.VideoCapture(self.camera_index)
        
def main():
    # Create the root window
    root = tk.Tk()
    root.title("Makeup Virtual Try-On")
    root.resizable(False, False)  # Disable resizing
    
    # Show startup message
    print("Virtual Makeup Application started")
    print("Press 'k' key to switch between cameras")
    
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