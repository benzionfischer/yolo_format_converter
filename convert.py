from ultralytics import YOLO

model = YOLO("yolo11n_custom_200_epoches_CPU_510_images_imgsz_320.pt")

# Export the model to TFLite format
model.export(format="tflite", int8=True, imgsz=320)

# after convertion to quantized tflite format, compile it for google coral:
# https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb?authuser=1#scrollTo=itv0Kj7N6ALw