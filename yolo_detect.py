import os
import sys
import glob
import time
import cv2
import numpy as np
import math
import requests
from ultralytics import YOLO

# === SETUP YOUR CONFIG HERE ===
model_path = "my_model.pt"
img_source = "usb0"  # Can be: "usb0", a path to image, folder, or video
min_thresh = 0.5
user_res = "640x480"
record = False

# === Load model ===
if not os.path.exists(model_path):
    print("❌ ERROR: YOLO model not found.")
    sys.exit()

model = YOLO(model_path, task='detect')
labels = model.names

# === Source detection ===
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mp4', '.mov', '.mkv']

source_type = ''
usb_idx = 0

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1]
    source_type = 'image' if ext.lower() in img_ext_list else 'video' if ext.lower() in vid_ext_list else ''
elif img_source.startswith("usb"):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print("❌ Invalid source.")
    sys.exit()

# === Resolution ===
resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print("❌ Invalid resolution format. Use 'WIDTHxHEIGHT'.")
        sys.exit()

# === Video/Camera Setup ===
if source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(usb_idx if source_type == 'usb' else img_source, cv2.CAP_DSHOW)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)
    if not cap.isOpened():
        print("❌ Failed to open video/camera source.")
        sys.exit()

if record:
    if source_type not in ['video', 'usb']:
        print("❌ Recording only supported for video/camera.")
        sys.exit()
    if not resize:
        print("❌ Please set resolution to enable recording.")
        sys.exit()
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# === Colors for bounding boxes ===
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241)]

# === Frame loop ===
imgs_list = []
if source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'image':
    imgs_list = [img_source]

img_count = 0
fps_log = []

while True:
    t_start = time.time()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print("✅ Done with all images.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print("✅ Done with video/camera.")
            break
    else:
        print("❌ Unsupported source type.")
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    clothes_detected = []

    for box in detections:
        conf = box.conf.item()
        if conf < min_thresh:
            continue

        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        class_id = int(box.cls.item())
        class_name = labels[class_id]
        color = bbox_colors[class_id % len(bbox_colors)]

        # Draw bounding box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_name}: {int(conf*100)}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin + 5), color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        clothes_detected.append(class_name)

    fps = 1.0 / (time.time() - t_start)
    fps_log.append(fps)
    if len(fps_log) > 100:
        fps_log.pop(0)

    if source_type in ['video', 'usb']:
        cv2.putText(frame, f"FPS: {np.mean(fps_log):.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Clothing Detection", frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video', 'usb'] else 0)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('snapshot.png', frame)
    elif key == ord('s'):
        cv2.waitKey()

# === Cleanup ===
print(f"📊 Avg FPS: {np.mean(fps_log):.2f}")
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

unique_clothes = set(clothes_detected)
print(unique_clothes)

# Clothing temperature guidelines
clothing_temp_ranges = {
    "Jacket": {"min_temp": -1 * math.inf, "max_temp": 15},
    "Jeans": {"min_temp": 5, "max_temp": 30},
    "Jogger": {"min_temp": 5, "max_temp": 25},
    "Polo": {"min_temp": 18, "max_temp": 32},
    "Shirt": {"min_temp": 16, "max_temp": 28},
    "Short": {"min_temp": 22, "max_temp": 40},
    "T-Shirt": {"min_temp": 18, "max_temp": math.inf},
    "Trouser": {"min_temp": 5, "max_temp": 30}
}


# === Get current location using IP and temperature from OpenWeather ===
openweather_api_key = "2803f0a97dc5858a4a10a7d80bbc63f2"

try:
    # Get user's approximate GPS coordinates from IP
    loc_response = requests.get("https://ipinfo.io/json")
    loc_data = loc_response.json()

    # Default to Tel Aviv if any key is missing
    if "loc" in loc_data:
        lat, lon = loc_data["loc"].split(",")
    else:
        lat, lon = "32.0853", "34.7818"  # Tel Aviv coordinates

    city_name = loc_data.get("city", "Tel Aviv")

    # Get weather data using OpenWeather API
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
    weather_response = requests.get(url)
    weather_data = weather_response.json()

    if weather_response.status_code == 200:
        current_temp = weather_data['main']['temp']
        print(f"\n🌡️ Current temperature in {city_name}: {current_temp}°C")
        for item in unique_clothes:
            temp_range = clothing_temp_ranges.get(item)
            if temp_range:
                if current_temp < temp_range["min_temp"]:
                    print(f"⚠️ {item} is too light for {current_temp}°C. Consider dressing warmer.")
                elif current_temp > temp_range["max_temp"]:
                    print(f"🥵 {item} may be too warm for {current_temp}°C. Consider dressing lighter.")
                else:
                    print(f"✅ {item} is appropriate for {current_temp}°C.")
    else:
        print("⚠️ Failed to get weather data:", weather_data.get("message"))

except Exception as e:
    print("⚠️ Could not locate user, defaulting to Tel Aviv.")
    try:
        # Use default Tel Aviv as fallback
        lat, lon = "32.0853", "34.7818"
        city_name = "Tel Aviv"
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
        weather_response = requests.get(url)
        weather_data = weather_response.json()

        if weather_response.status_code == 200:
            current_temp = weather_data['main']['temp']
            print(f"\n🌡️ Current temperature in Tel Aviv: {current_temp}°C")
            for item in unique_clothes:
                temp_range = clothing_temp_ranges.get(item)
                if temp_range:
                    if current_temp < temp_range["min_temp"]:
                        print(f"⚠️ {item} is too light for {current_temp}°C. Consider dressing warmer.")
                    elif current_temp > temp_range["max_temp"]:
                        print(f"🥵 {item} may be too warm for {current_temp}°C. Consider dressing lighter.")
                    else:
                        print(f"✅ {item} is appropriate for {current_temp}°C.")
        else:
            print("⚠️ Failed to get weather data:", weather_data.get("message"))
    except Exception as e2:
        print(f"❌ Final weather lookup failed: {e2}")


    