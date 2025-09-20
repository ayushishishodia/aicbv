import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import threading
import pyttsx3
import pandas as pd

# ----------------------- Streamlit Config -----------------------
st.set_page_config(page_title="Public Crowd Alert", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: lavender;}
    .stButton>button {background-color: red; color: black; font-weight:bold;}
    .stTextInput>div>input {background-color: #FFFFFF; color: black; font-size:16px;}
    h1, h2, h3 {color: black;}
    </style>
    """, unsafe_allow_html=True
)
st.title("🧑‍🤝‍🧑 Public Crowd Alert")
st.write("Monitor crowd density, get alerts, view statistics, and chat with your Safety Advisor.")

# ----------------------- Location Input -----------------------
location = st.text_input("Enter the current location or GPS coordinates:")
st.session_state.location = location if location else "Unknown Location"

# ----------------------- Upload OR Live Option -----------------------
st.subheader("Choose Input Source:")
video_file = st.file_uploader("Upload a recorded video", type=["mp4", "avi", "mov"])
live_detection = st.button("📷 Start Live Camera Detection")

# ----------------------- Settings -----------------------
CROWD_THRESHOLD = 10
CONFIDENCE_THRESHOLD = 0.3
ROLLING_FRAMES = 5
PREDICT_SECONDS = 30  # Predict next ~30 frames

# ----------------------- Session state -----------------------
if 'current_average' not in st.session_state: st.session_state.current_average = 0
if 'max_count' not in st.session_state: st.session_state.max_count = 0
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'alert_triggered' not in st.session_state: st.session_state.alert_triggered = False
if 'predicted_count' not in st.session_state: st.session_state.predicted_count = 0
if 'crowd_stats' not in st.session_state: st.session_state.crowd_stats = pd.DataFrame(columns=['frame', 'count', 'predicted'])

# ----------------------- Text-to-Speech -----------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ----------------------- Chatbot Function -----------------------
def chatbot_response(question):
    q = question.lower()
    ca = st.session_state.current_average
    mc = st.session_state.max_count
    loc = st.session_state.location
    pc = st.session_state.predicted_count

    if any(word in q for word in ['hi','hello','hey']):
        return f"Hello! I am your Safety Advisor at {loc}. Ask me about crowd safety or predicted buildup."
    if any(word in q for word in ['bye','goodbye','see you']):
        return "Goodbye! Stay safe and keep monitoring the crowd."
    if "safe" in q or "crowd" in q or "overcrowded" in q:
        if ca > CROWD_THRESHOLD:
            return f"🚨 Current crowd is {ca}, exceeds safe limits at {loc}!"
        elif ca > CROWD_THRESHOLD*0.7:
            return f"⚠️ Crowd is approaching limit ({ca}) at {loc}."
        else:
            return f"✅ Crowd is safe ({ca}) at {loc}. Max observed: {mc}. Predicted next: {pc}"
    if "max" in q:
        return f"Maximum people counted so far: {mc} at {loc}."
    if "forecast" in q or "predict" in q:
        return f"Predicted crowd in next few seconds: {pc}"
    if "how crowded" in q or "looks" in q:
        if ca < CROWD_THRESHOLD*0.5:
            return "Crowd looks light."
        elif ca < CROWD_THRESHOLD*0.7:
            return "Crowd looks moderate."
        else:
            return "Crowd looks heavy."
    return "ℹ️ I am your Safety Advisor. Ask about crowd safety or just chat with me!"

# ----------------------- Sidebar Chat -----------------------
st.sidebar.title("🗨️ Safety Advisor Chat")
user_input = st.sidebar.text_input("Type your message here:")
if st.sidebar.button("Send") and user_input.strip() != "":
    response = chatbot_response(user_input)
    st.session_state.chat_history.append((user_input, response))
for q, a in st.session_state.chat_history:
    st.sidebar.markdown(f"**You:** {q}")
    st.sidebar.markdown(f"**Advisor:** {a}")

# ----------------------- Detection Function -----------------------
def run_detection(cap, live=False):
    model = YOLO("yolov8n.pt")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live else 0
    progress_bar = st.progress(0) if not live else None
    stframe = st.empty()
    status_text = st.empty()
    heatmap_acc = None
    recent_counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        annotated_frame = results[0].plot()
        current_count = sum(1 for r in results[0].boxes if int(r.cls[0])==0)
        recent_counts.append(current_count)
        if len(recent_counts) > ROLLING_FRAMES: recent_counts.pop(0)
        current_average = int(sum(recent_counts)/len(recent_counts))
        st.session_state.current_average = current_average
        st.session_state.max_count = max(st.session_state.max_count, current_average)

        # -------------------- Crowd Forecast --------------------
        slope = (recent_counts[-1] - recent_counts[0]) / len(recent_counts) if len(recent_counts) > 1 else 0
        forecast_counts = [int(current_average + slope*i) for i in range(1, PREDICT_SECONDS+1)]
        st.session_state.predicted_count = forecast_counts[-1]

        # Update crowd stats for graph
        st.session_state.crowd_stats = pd.concat([st.session_state.crowd_stats,
                                                 pd.DataFrame({'frame':[cap.get(cv2.CAP_PROP_POS_FRAMES) if not live else len(st.session_state.crowd_stats)+1],
                                                               'count':[current_average],
                                                               'predicted':[st.session_state.predicted_count]})],
                                                 ignore_index=True)

        # -------------------- Alert Logic --------------------
        if max(forecast_counts) > CROWD_THRESHOLD or current_average > CROWD_THRESHOLD:
            alert = "🚨 Overcrowding Alert!"
            color = "red"
            if not st.session_state.alert_triggered:
                def speak_alert():
                    engine.say(f"Alert! Crowd is exceeding safe limits at {st.session_state.location}. Please evacuate immediately!")
                    engine.runAndWait()
                threading.Thread(target=speak_alert, daemon=True).start()
                st.session_state.alert_triggered = True
        elif current_average > CROWD_THRESHOLD*0.7:
            alert = "⚠️ Approaching Limit"
            color = "orange"
        else:
            alert = "✅ Safe"
            color = "green"

        # -------------------- Status --------------------
        status_text.markdown(
            f"<h3 style='color:{color}'>Location: {st.session_state.location} | People Count: {current_average} | Max Count: {st.session_state.max_count} | Predicted Next: {st.session_state.predicted_count} | Status: {alert}</h3>",
            unsafe_allow_html=True
        )

        # -------------------- Heatmap Overlay --------------------
        heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        people_positions = [(int((b.xyxy[0][0]+b.xyxy[0][2])/2), int((b.xyxy[0][1]+b.xyxy[0][3])/2)) 
                           for b in results[0].boxes if int(b.cls[0])==0]
        for (cx, cy) in people_positions: cv2.circle(heatmap_layer, (cx, cy), 50, 1, -1)
        heatmap_acc = heatmap_layer if heatmap_acc is None else cv2.addWeighted(heatmap_acc, 0.9, heatmap_layer, 0.1, 0)
        heatmap_norm = np.uint8(255*heatmap_acc/np.max(heatmap_acc+1e-5))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlayed_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_color, 0.3, 0)

        stframe.image(overlayed_frame, channels="BGR")
        if progress_bar:
            progress_bar.progress(min(int(cap.get(cv2.CAP_PROP_POS_FRAMES)/frame_count*100),100))

    cap.release()
    st.success("✅ Detection Completed!")

    # -------------------- Crowd Statistics Graph --------------------
    st.subheader("📊 Crowd Statistics Over Time")
    st.line_chart(st.session_state.crowd_stats.set_index('frame')[['count','predicted']])

# ----------------------- Run Detection -----------------------
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    run_detection(cap, live=False)

if live_detection:
    cap = cv2.VideoCapture(0)  # open webcam
    run_detection(cap, live=True)
