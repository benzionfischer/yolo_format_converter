from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Export the model to TFLite format
model.export(format="tflite", imgsz=320, int8=True)

# after convertion to quantized tflite format, compile it for google coral:
# https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb?authuser=1#scrollTo=itv0Kj7N6ALw