from ultralytics import YOLO

model = YOLO("yolov8n_custom_200_epoches_CPU_510_images.pt")

# Export the model to TFLite format
model.export(format="tflite", imgsz=240, int8=True)

# after convertion to quantized tflite format, compile it for google coral:
# https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb?authuser=1#scrollTo=itv0Kj7N6ALw