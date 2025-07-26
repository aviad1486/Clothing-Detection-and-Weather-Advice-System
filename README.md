# 🧥 Clothing Detection and Weather Advice System

This project is a real-time clothing detection system using a YOLO model. It detects specific clothing items via webcam and provides smart dressing advice based on current weather at your location — with automatic fallback to Tel Aviv if GPS lookup fails.

## 👕 Supported Clothing Classes
The model is trained to detect the following clothing types:
- Jacket
- Jeans
- Jogger
- Polo
- Shirt
- Short
- T-Shirt
- Trouser


## 📌 Project Workflow

### 1. 🔖 Data Labeling
Clothing images were labeled using [Label Studio](https://labelstud.io/) — a powerful open-source data annotation tool. After annotation, Label Studio exported the labeled data in a `data.zip` file containing all annotations and images.

### 2. 🧠 Model Training
The YOLO model was trained using a script called `Train_YOLO_Models.ipynb`. It loads the labeled dataset, configures training parameters, and produces a YOLOv11-compatible `.pt` model (`my_model.pt`).

### 3. 🎥 Real-Time Detection
A custom script `yolo_detect.py` performs:
- Real-time detection via USB webcam
- Drawing bounding boxes around detected clothing items
- Calculating FPS for performance feedback
- Saving snapshots with a keypress

### 4. 🌍 GPS-Based Weather Dressing Advice
After detection, the script:
- Automatically detects your approximate location using [ipinfo.io](https://ipinfo.io/)
- Fetches the current temperature using the [OpenWeather API](https://openweathermap.org/)
- Falls back to **Tel Aviv** if location can't be determined
- Compares detected clothing items against temperature ranges
- Prints real-time dressing advice based on weather

### 5. 🧠 Train on Your Own Dataset

You can easily train the model on your own custom dataset!

### Steps:
1. **Label your images** using [Label Studio](https://labelstud.io/), an open-source data annotation tool.
2. Export the labeled data as a `data.zip` file (containing both images and annotations).
3. Run and follow the instruction of the training script:
   ```bash
   python Train_YOLO_Models.ipynb
## 🛠️ Requirements

- Python 3.8+
- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- OpenCV
- NumPy
- Requests

Install dependencies:

```bash
pip install ultralytics opencv-python numpy requests
```
## 📸 Examples

<p align="center">
  <img src="Examples/T-Shirt-Detection.png" alt="Clothing Detection Example 1" width="400"/>
  <img src="Examples/JacketDetection.png" alt="Clothing Detection Example 2" width="400"/>
</p>

<ul>
  <li>🎞️ <strong>Example video</strong>: <a href="https://www.canva.com/design/DAGmmFqRazw/x_IlixjudVelwOI7GTAHoQ/watch?utm_content=DAGmmFqRazw&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hc573914370" target="_blank">Watch here</a></li>
  <li>📢 <strong>Marketing video</strong>: <a href="https://www.canva.com/design/DAGmCjYZ5Wg/IJZgna6FLM65FF57lTTqtg/watch?utm_content=DAGmCjYZ5Wg&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h7eb44288b5" target="_blank">Watch here</a></li>
  <li>📊 <strong>Project presentation</strong>: <a href="https://www.canva.com/design/DAGswMsshQg/2wFzSfMRrHfGbxj_zq9sFQ/watch?utm_content=DAGswMsshQg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton" target="_blank">Watch here</a></li>
</ul>
