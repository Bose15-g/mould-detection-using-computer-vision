import os
import threading
import time
from tkinter import filedialog
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

# ------------------------------------------------------------------------
# Global App Configuration & State
# ------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Industrial Fill / No Fill Detection System")
app.geometry("1200x750")

# Application control variables
video_path = ""
is_detecting = False
detection_thread = None

# UI Parameter variables linked directly to inputs
source_mode_var = ctk.StringVar(value="File")  # "File" or "Live Camera"
camera_index_var = ctk.StringVar(value="0")  # Index or RTSP stream URI
# Threshold UI-la ilanaalum, background logic-kaga 60-la set pannirukom
threshold_var = ctk.IntVar(value=60) 
rotation_var = ctk.IntVar(value=0)
brightness_var = ctk.IntVar(value=220)


# ------------------------------------------------------------------------
# Core Image Processing Engine (OpenCV)
# ------------------------------------------------------------------------
def process_video_stream():
    """Runs in a background thread to process frames without freezing the UI."""
    global is_detecting, video_path

    # Determine input source dynamically based on UI switch
    if source_mode_var.get() == "Live Camera":
        cam_input = camera_index_var.get()
        if cam_input.isdigit():
            source = int(cam_input)
        else:
            source = cam_input
    else:
        source = video_path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        is_detecting = False
        app.after(0, update_ui_state_stopped)
        return

    # ==================================
    # FILL VALIDATION VARIABLES
    # ==================================
    fill_counter = 0
    required_frames = 5

    while is_detecting:
        ret, frame = cap.read()

        if not ret:
            if source_mode_var.get() == "File":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                time.sleep(0.1)
                continue

        # 1. Apply Dynamic Rotation
        angle = rotation_var.get()
        if angle != 0:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame = cv2.warpAffine(frame, rotation_matrix, (w, h))

        # ==================================
        # 2. IMPROVED HSV / BRIGHTNESS LOGIC
        # ==================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, v_channel = cv2.split(hsv)

        # Apply blur to remove minor camera grain/noise
        v_channel = cv2.medianBlur(v_channel, 5)

        thresh_value = brightness_var.get()
        _, molten_mask = cv2.threshold(
            v_channel, thresh_value, 255, cv2.THRESH_BINARY
        )

        # Apply Morphology (Noise cancellation mathiri - removes small false dots)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        molten_mask = cv2.morphologyEx(molten_mask, cv2.MORPH_OPEN, kernel)
        molten_mask = cv2.morphologyEx(molten_mask, cv2.MORPH_CLOSE, kernel)

        # Keep ONLY the largest molten region (Ignores background glare/reflections)
        contours, _ = cv2.findContours(molten_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        binary_mask = np.zeros_like(molten_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(binary_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # 3. Mould Circle Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=100,
            maxRadius=500,
        )

        fill_percentage = 0.0

        if circles is not None:
            circles = np.round(circles[0]).astype(int)

            # Isolate the largest detected circular mould
            x, y, r = max(circles, key=lambda c: c[2])

            # Create a blank mask matching the binary mask's layout
            mould_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.circle(mould_mask, (x, y), r, 255, -1)

            # Bitwise AND to find molten metal specifically INSIDE the mould
            molten_inside = cv2.bitwise_and(binary_mask, mould_mask)

            mould_pixels = cv2.countNonZero(mould_mask)
            molten_pixels = cv2.countNonZero(molten_inside)

            if mould_pixels > 0:
                fill_percentage = (molten_pixels / mould_pixels) * 100
        else:
            cv2.putText(
                frame,
                "MOULD NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # ==================================
        # CONSECUTIVE FRAME LOGIC
        # ==================================
        target_threshold = threshold_var.get()

        if fill_percentage >= target_threshold:
            fill_counter += 1
        else:
            fill_counter = 0

        if fill_counter >= required_frames:
            status_text = "STATUS : FILL (PASSED)"
            status_color = "#4BB543"
        else:
            status_text = "STATUS : NO FILL (DEFECT)"
            status_color = "#FF3333"

        # 4. Update the UI Elements safely from the thread
        app.after(0, lambda t=status_text, c=status_color: update_metrics_ui(t, c))

        # 5. Prepare and render the live preview frame
        preview_frame = cv2.resize(frame, (800, 500))
        preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(preview_frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        video_frame.configure(image=img_tk, text="")
        video_frame.image = img_tk  # Avoid garbage collection

        time.sleep(0.03)

    cap.release()


# ------------------------------------------------------------------------
# UI Event Handlers & Core Logic
# ------------------------------------------------------------------------
def upload_video():
    global video_path
    path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if path:
        video_path = path
        video_label.configure(text=os.path.basename(path))
        source_segmented.set("File")


def handle_source_change(value):
    """Dynamically shows/hides context controls based on selection."""
    if value == "Live Camera":
        camera_input_label.pack(after=source_segmented, pady=(5, 0))
        camera_input_entry.pack(after=camera_input_label, fill="x", padx=30, pady=(0, 10))
    else:
        camera_input_label.pack_forget()
        camera_input_entry.pack_forget()


def toggle_detection():
    global is_detecting, detection_thread

    if not is_detecting:
        if source_mode_var.get() == "File" and not video_path:
            status.configure(
                text="STATUS : SELECT VIDEO FIRST", text_color="orange"
            )
            return

        is_detecting = True
        start_btn.configure(text="STOP DETECTION", fg_color="#d9534f")
        status.configure(text="STATUS : RUNNING", text_color="cyan")

        detection_thread = threading.Thread(
            target=process_video_stream, daemon=True
        )
        detection_thread.start()
    else:
        is_detecting = False
        update_ui_state_stopped()


def update_ui_state_stopped():
    """Resets UI interactive elements back to standby state."""
    start_btn.configure(text="START DETECTION", fg_color=["#3B8ED0", "#1F6AA5"])
    status.configure(text="STATUS : STOPPED", text_color="yellow")


def update_metrics_ui(status_text, status_color):
    """Receives calculated status from thread and updates UI."""
    status.configure(text=status_text, text_color=status_color)


# ------------------------------------------------------------------------
# UI Layout Construction
# ------------------------------------------------------------------------
left_panel = ctk.CTkFrame(app, width=300)
left_panel.pack(side="left", fill="y", padx=10, pady=10)

title = ctk.CTkLabel(
    left_panel,
    text="Fill / No Fill\nMould Detection",
    font=("Arial", 24, "bold"),
)
title.pack(pady=20)

# Input Source Toggle Controls
ctk.CTkLabel(left_panel, text="Input Source Selection").pack(pady=(10, 0))
source_segmented = ctk.CTkSegmentedButton(
    left_panel,
    values=["File", "Live Camera"],
    variable=source_mode_var,
    command=handle_source_change,
)
source_segmented.pack(fill="x", padx=20, pady=5)

upload_btn = ctk.CTkButton(
    left_panel, text="Upload Source Video", command=upload_video
)
upload_btn.pack(pady=10)

video_label = ctk.CTkLabel(
    left_panel, text="No Video Selected", font=("Arial", 12, "italic")
)
video_label.pack()

camera_input_label = ctk.CTkLabel(left_panel, text="Webcam Index or RTSP URL")
camera_input_entry = ctk.CTkEntry(left_panel, textvariable=camera_index_var)

# Dynamic Adjusters & Sliders Section
ctk.CTkLabel(left_panel, text="Camera Rotation Orientation (0-360°)").pack(
    pady=(25, 0)
)
rotation_slider = ctk.CTkSlider(
    left_panel, from_=0, to=360, variable=rotation_var
)
rotation_slider.pack(fill="x", padx=20)

ctk.CTkLabel(left_panel, text="HSV Value/Brightness Threshold").pack(
    pady=(20, 0)
)
brightness_slider = ctk.CTkSlider(
    left_panel, from_=0, to=255, variable=brightness_var
)
brightness_slider.pack(fill="x", padx=20)

# Toggle Action Button
start_btn = ctk.CTkButton(
    left_panel, text="START DETECTION", height=45, command=toggle_detection
)
start_btn.pack(side="bottom", fill="x", padx=20, pady=40)

# Right Panel Display Metrics & Visualization Frame
right_panel = ctk.CTkFrame(app)
right_panel.pack(side="right", expand=True, fill="both", padx=10, pady=10)

video_frame = ctk.CTkLabel(
    right_panel,
    text="Industrial Feed Input Standby",
    width=800,
    height=500,
    fg_color="#1e1e1e",
)
video_frame.pack(pady=20)

# Only Status is displayed now
status = ctk.CTkLabel(
    right_panel,
    text="STATUS : STANDBY AWAITING INPUT",
    font=("Arial", 22, "bold"),
    text_color="yellow",
)
status.pack(pady=30) 

app.mainloop()