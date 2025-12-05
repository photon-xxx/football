import os
from ultralytics import YOLO 

# 获取项目根目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)

model = YOLO(os.path.join(project_root, 'Storage', 'models', 'yolo', 'best.pt'))

input_video = os.path.join(project_root, 'IO', 'input_videos', '08fd33_4.mp4')
results = model.predict(input_video, save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)