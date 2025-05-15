from typing_extensions import final
from flask import Flask, request, send_file, jsonify, send_from_directory
from pyngrok import ngrok
import os
import uuid
import cv2
from moviepy.editor import ImageSequenceClip
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# 创建目录
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/videos", exist_ok=True)

# 建立 ngrok 隧道
public_url = ngrok.connect(5000)
print(" * Ngrok tunnel URL:", public_url)

# 加载 YOLOv8 模型
yolo_model = YOLO("yolov8n.pt")

# 上传图片接口
@app.route('/upload/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # 保存原图
    filename = str(uuid.uuid4()) + "_" + file.filename
    path = os.path.join("uploads/images", filename)
    file.save(path)

    # 读取图片并进行目标检测
    img = cv2.imread(path)
    results = yolo_model(img)[0]
    annotated = results.plot()

    # 保存带框图片
    result_filename = "result_" + filename
    result_path = os.path.join("uploads/images", result_filename)
    cv2.imwrite(result_path, annotated)

    # 返回带框图片路径
    return f"uploads/images/{result_filename}", 200


# 上传视频接口并进行目标检测处理
@app.route('/upload/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # 生成唯一文件名
    video_id = str(uuid.uuid4())
    video_path = f"uploads/videos/{video_id}.mp4"
    out_path = f"uploads/videos/{video_id}_out.mp4"
    file.save(video_path)

    # 读取视频并检测
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame)[0]
        annotated = results.plot()
        frames.append(annotated)
    cap.release()

    # 保存结果视频
    clip = ImageSequenceClip(frames, fps=20)
    clip.write_videofile(out_path, codec='libx264')

    return f"{out_path}", 200

# 提供图片访问接口
@app.route('/uploads/images/<filename>')
def get_image(filename):
    return send_from_directory('uploads/images', filename)

# 提供视频访问接口
@app.route('/uploads/videos/<filename>')
def get_video(filename):
    return send_from_directory('uploads/videos', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # 支持 Railway 自动端口
    app.run(host='0.0.0.0', port=port)

