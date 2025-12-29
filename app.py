import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO, RTDETR
import tempfile
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import torch
from collections import deque

# Detect and set device
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, "CUDA"
    # elif torch.backends.mps.is_available(): # Mac용 코드 (없어도 무방)
    #     device = torch.device("mps")
    #     return device, "MPS"
    else:
        device = torch.device("cpu")
        return device, "CPU"

# Define colors for each class (BGR format)
CLASS_COLORS = {
    'Person': (255, 255, 255),       # White
    'gloves': (0, 128, 0),           # Green
    'goggles': (0, 128, 0),          # Green
    'helmet': (0, 128, 0),           # Green
    'no_gloves': (0, 0, 255),        # Red
    'no_goggle': (0, 0, 255),        # Red
    'no_helmet': (0, 0, 255)         # Red
}

# Load models based on selection
@st.cache_resource
def load_yolo_model(model_path):
    device, device_name = get_device()
    model = YOLO(model_path)
    model.to(device)
    print(f"YOLO model loaded on device: {device}")
    return model, device, device_name

@st.cache_resource
def load_rtdetr_model(model_path):
    device, device_name = get_device()
    model = RTDETR(model_path)
    model.to(device)
    print(f"RT-DETR model loaded on device: {device}")
    return model, device, device_name

st.title("PPE Detection System")
st.write("Detect whether people are wearing proper personal protective equipment")

# Model selection in sidebar (outside of modes)
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio(
    "Choose Detection Model:",
    ["YOLOv11", "RT-DETR v1", "YOLOv11 vs RT-DETR v1 (Image only)"],
    help="Select which model to use. Comparison mode only works for image upload."
)

# --- [수정된 부분] ---
# Windows 호환을 위해 상대 경로로 변경했습니다.
# 이 .py 파일과 같은 폴더에 모델 파일이 있어야 합니다.
yolo_model_path = 'weights/yolo_model.pt'
rtdetr_model_path = 'weights/rtdetr_model.pt'
# --- [수정 완료] ---


# Load selected model(s)
if model_type == "YOLOv11":
    model, device, device_name = load_yolo_model(yolo_model_path)
    st.sidebar.info(f"📦 Loaded: YOLOv11")
    comparison_mode = False
elif model_type == "RT-DETR v1":
    model, device, device_name = load_rtdetr_model(rtdetr_model_path)
    st.sidebar.info(f"📦 Loaded: RT-DETR v1")
    comparison_mode = False
else:  # Comparison mode
    yolo_model, yolo_device, yolo_device_name = load_yolo_model(yolo_model_path)
    rtdetr_model, rtdetr_device, rtdetr_device_name = load_rtdetr_model(rtdetr_model_path)
    model = yolo_model  # Default for non-image modes
    device = yolo_device
    device_name = yolo_device_name
    st.sidebar.info(f"📦 Loaded: Both models")
    comparison_mode = True

# Display device info in small grey text
st.sidebar.markdown(f"<p style='font-size:11px; color:#666666;'>{device_name} utilized</p>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Function to draw custom colored bounding boxes
def draw_detections(frame, results, detection_model=None):
    # Use provided model or fall back to global model
    if detection_model is None:
        detection_model = model
    
    # Convert RGB to BGR if needed (PIL images are RGB, OpenCV is BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Check if it's RGB (from PIL) and convert to BGR for OpenCV
        if isinstance(frame, np.ndarray) and frame.dtype == np.uint8:
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            annotated_frame = frame.copy()
    else:
        annotated_frame = frame.copy()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        label = detection_model.names[cls]
        
        # Get color for this class
        color = CLASS_COLORS.get(label, (0, 255, 0))  # Default to green if not found
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f'{label} {conf:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Convert back to RGB for display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ["Image Upload", "Video Upload", "Webcam (Real-time)"])

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if comparison_mode:
            # Comparison mode: run both models
            st.subheader("Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**YOLOv11 Detection**")
                with torch.no_grad():
                    yolo_results = yolo_model(img_array, device=yolo_device)
                yolo_annotated = draw_detections(img_array, yolo_results, yolo_model)
                st.image(yolo_annotated, caption="YOLOv11 Results", use_container_width=True)
                
                st.write("Detections:")
                for box in yolo_results[0].boxes:
                    class_name = yolo_model.names[int(box.cls)]
                    confidence = float(box.conf)
                    st.write(f"- {class_name} ({confidence:.2f})")
            
            with col2:
                st.write("**RT-DETR v1 Detection**")
                with torch.no_grad():
                    rtdetr_results = rtdetr_model(img_array, device=rtdetr_device)
                rtdetr_annotated = draw_detections(img_array, rtdetr_results, rtdetr_model)
                st.image(rtdetr_annotated, caption="RT-DETR v1 Results", use_container_width=True)
                
                st.write("Detections:")
                for box in rtdetr_results[0].boxes:
                    class_name = rtdetr_model.names[int(box.cls)]
                    confidence = float(box.conf)
                    st.write(f"- {class_name} ({confidence:.2f})")
        
        else:
            # Single model mode
            with torch.no_grad():
                results = model(img_array, device=device)
            
            # Draw custom colored bounding boxes
            annotated_img = draw_detections(img_array, results)
            
            # Display results (image is now in RGB format)
            st.image(annotated_img, caption="Detection Results")
            
            # --- [버그 수정] ---
            # 아래 Detections 블록이 원래 밖에 빠져있어 비교 모드에서 오류가 발생했습니다.
            # else: 블록 안으로 이동시켰습니다.
            st.subheader("Detections:")
            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)
                st.write(f"- {class_name} (Confidence: {confidence:.2f})")
            # --- [수정 완료] ---
        
elif mode == "Video Upload":
    if comparison_mode:
        st.warning("Comparison mode is only available for Image Upload.")
    
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        input_path = "input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.success("✅ Video uploaded successfully!")
        
        # Processing options
        st.write("**Processing Options:**")
        process_every_n = st.selectbox(
            "Process every N frames (1 = all frames, higher = faster processing):",
            [1, 2, 3, 5, 10],
            index=0
        )
        
        # Process video button
        if st.button("🎬 Process Video with PPE Detection"):
            temp_output = "temp_output.avi"
            output_path = "output_video.mp4"
            
            # Open video
            cap = cv2.VideoCapture(input_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # First save as AVI (more reliable)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            last_detection_result = None
            
            st.info(f"🎥 Processing {total_frames} frames at {fps} FPS using {device_name}...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection on selected frames with explicit device
                if frame_count % process_every_n == 0:
                    # BGR 프레임을 RGB로 변환하여 모델에 전달 (ultralytics 모델은 RGB를 선호)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        results = model(frame_rgb, device=device)
                    
                    # draw_detections 함수는 BGR 프레임을 기본으로 받음
                    annotated_frame = draw_detections(frame, results)
                    last_detection_result = annotated_frame
                else:
                    # Use last detection result or original frame
                    annotated_frame = last_detection_result if last_detection_result is not None else frame
                
                # Write frame to output video (draw_detections이 RGB로 반환하므로 BGR로 다시 변환)
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
            
            # Release resources
            cap.release()
            out.release()
            
            # Convert to web-compatible MP4 using ffmpeg if available
            status_text.text("Converting to web-compatible format...")
            try:
                import subprocess
                subprocess.run([
                    'ffmpeg', '-i', temp_output, '-c:v', 'libx264', 
                    '-preset', 'fast', '-crf', '22', '-y', output_path
                ], check=True, capture_output=True)
                os.remove(temp_output)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg not available, try direct H.264 encoding
                status_text.text("ffmpeg not found, trying alternative method...")
                cap2 = cv2.VideoCapture(temp_output)
                fourcc2 = cv2.VideoWriter_fourcc(*'avc1')
                out2 = cv2.VideoWriter(output_path, fourcc2, fps, (width, height))
                
                while cap2.isOpened():
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    out2.write(frame)
                
                cap2.release()
                out2.release()
                os.remove(temp_output)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Video processing complete!")
            
            # Store processed video path in session state
            st.session_state.processed_video_path = output_path
        
        # Display processed video
        if 'processed_video_path' in st.session_state and os.path.exists(st.session_state.processed_video_path):
            st.write("---")
            st.write("**Processed Video with PPE Detection:**")
            
            # Display video with native HTML5 controls (play, pause, seek, speed control)
            st.video(st.session_state.processed_video_path)
            
            st.info("💡 Tip: Use the video controls to play, pause, seek, and adjust playback speed. The video plays smoothly at full speed with all detections visible!")
            
            # Download button
            with open(st.session_state.processed_video_path, "rb") as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file,
                    file_name="ppe_detection_output.mp4",
                    mime="video/mp4"
                )

elif mode == "Webcam (Real-time)":
    if comparison_mode:
        st.warning("Comparison mode is only available for Image Upload.")
        
    st.write("Real-time webcam detection using WebRTC")
    
    # Define the video processor class
    class PPEDetector(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # BGR 프레임을 RGB로 변환하여 모델에 전달
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run detection with explicit device and no_grad for efficiency
            with torch.no_grad():
                results = model(img_rgb, device=device)
            
            # draw_detections 함수는 BGR 프레임을 기본으로 받음
            annotated_img = draw_detections(img, results)
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
    
    # Start webcam stream
    webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=PPEDetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )