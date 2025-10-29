import os
import cv2
# 让 Python 走你的 VPN 本地代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"


# # import a utility function for loading Roboflow models
from inference import get_model

# define the image url to use for inference
image = "E:\Soccer\\football\\football_analysis-main\IO\input_videos\screen.jpg"

# load a pre-trained yolov8n model

cap = cv2.VideoCapture("/IO/input_videos/test.mp4")
out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()
    if not ret:
        break


model = get_model(
    model_id="futvar-football-players-detection-dataset/7",
    api_key="CC10Uf8jM9o9q2ylp9ZL",  # 👈 替换成你自己的 key
)

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer("E:\Soccer\\football\\football_analysis-main\IO\input_videos\\test.mp4")

for pred in results.predictions:
    x1 = int(pred.x - pred.width / 2)
    y1 = int(pred.y - pred.height / 2)
    x2 = int(pred.x + pred.width / 2)
    y2 = int(pred.y + pred.height / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

out.write(frame)

cap.release()
out.release()
print("✅ 视频检测完成，保存为 output.mp4")
####################################################################################################################
# import onnxruntime as ort
# print(ort.get_device())
# print(ort.get_available_providers())
########################################################################################
# import onnxruntime as ort
#
# print("onnxruntime version:", ort.__version__)
# print("available providers:", ort.get_available_providers())
#
# sess = ort.InferenceSession(None)  # 仅创建 Session 环境
# print("default provider:", ort.get_device())
#
# ROBOFLOW_API_KEY="CC10Uf8jM9o9q2ylp9ZL"
# from inference import InferencePipeline
# from inference.core.interfaces.stream.sinks import render_boxes
#
# pipeline = InferencePipeline.init(
#     model_id="futvar-football-players-detection-dataset/7",
#     video_reference="E:/Soccer/football/football_analysis-main/IO/input_videos/08fd33_4.mp4", # Replace with the path to your video
#     on_prediction=render_boxes, # Function to run after each prediction
# )
# pipeline.start()
# pipeline.join()


#################################################################################################
# import cv2
#
# print("OpenCV 版本:", cv2.__version__)
# print("OpenCV 路径:", cv2.__file__)
#
# print("FONT_HERSHEY_SIMPLEX:", cv2.FONT_HERSHEY_SIMPLEX)
# print("COLOR_BGR2GRAY:", hasattr(cv2, 'COLOR_BGR2GRAY'))


###################################################################################################
#
# # 1. Import the InferencePipeline library
# from inference import InferencePipeline
# import cv2
#
# def my_sink(result, video_frame):
#     if result.get("output_image"): # Display an image from the workflow response
#         cv2.imshow("Workflow Image", result["output_image"].numpy_image)
#         cv2.waitKey(1)
#     # Do something with the predictions of each frame
#     print(result)
#
#
# # 2. Initialize a pipeline object
# pipeline = InferencePipeline.init_with_workflow(
#     api_key="CC10Uf8jM9o9q2ylp9ZL",
#     workspace_name="pig-m7pr8",
#     workflow_id="custom-workflow",
#     video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
#     max_fps=30,
#     on_prediction=my_sink
# )
#
# # 3. Start the pipeline and wait for it to finish
# pipeline.start()
# pipeline.join()
