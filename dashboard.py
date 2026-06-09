import streamlit as st
import cv2
from ultralytics import YOLO
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Traffic Management System",
    layout="wide"
)

st.title("🚦 AI Traffic Management System")
st.markdown("### Real-Time Vehicle Detection, Tracking and Congestion Prediction")

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO("yolov8m.pt")

video_path = "traffic.avi"

tracked_classes = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "person"
]

# =====================================================
# DASHBOARD PLACEHOLDERS
# =====================================================

frame_window = st.empty()

top1, top2, top3 = st.columns(3)

vehicle_count_box = top1.empty()
traffic_box = top2.empty()
signal_box = top3.empty()

breakdown_placeholder = st.empty()

# =====================================================
# VIDEO CAPTURE
# =====================================================

cap = cv2.VideoCapture(video_path)

unique_objects = set()

# =====================================================
# PROCESS VIDEO
# =====================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # ============================================
    # TRACK IDS
    # ============================================

    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for obj_id, cls in zip(ids, classes):

            class_name = model.names[int(cls)]

            if class_name in tracked_classes:
                unique_objects.add(
                    (int(obj_id), class_name)
                )

    # ============================================
    # UNIQUE COUNTS
    # ============================================

    counts = {}

    for _, class_name in unique_objects:
        counts[class_name] = counts.get(class_name, 0) + 1

    car_count = counts.get("car", 0)
    bus_count = counts.get("bus", 0)
    truck_count = counts.get("truck", 0)
    motorcycle_count = counts.get("motorcycle", 0)

    total_vehicles = (
        car_count
        + bus_count
        + truck_count
        + motorcycle_count
    )

    # ============================================
    # CONGESTION PREDICTION
    # ============================================

    if total_vehicles < 30:
        traffic = "LOW"
        green_time = 20

    elif total_vehicles < 60:
        traffic = "MEDIUM"
        green_time = 40

    else:
        traffic = "HIGH"
        green_time = 60

    # ============================================
    # DISPLAY VIDEO
    # ============================================

    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    frame_window.image(
        annotated_frame,
        channels="RGB",
        width="stretch"
    )

    # ============================================
    # MAIN METRICS
    # ============================================

    vehicle_count_box.metric(
        "🚗 Total Vehicles",
        total_vehicles
    )

    traffic_box.metric(
        "🚦 Traffic Density",
        traffic
    )

    signal_box.metric(
        "⏱ Green Signal",
        f"{green_time} sec"
    )

    # ============================================
    # VEHICLE BREAKDOWN
    # ============================================

    with breakdown_placeholder.container():

        st.markdown("## Vehicle Breakdown")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🚗 Cars", car_count)
        c2.metric("🚌 Buses", bus_count)
        c3.metric("🚚 Trucks", truck_count)
        c4.metric("🏍 Motorcycles", motorcycle_count)

    time.sleep(0.03)

# =====================================================
# END
# =====================================================

cap.release()

st.success("✅ Video Processing Complete")