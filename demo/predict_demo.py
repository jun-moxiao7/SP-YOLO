import os
from ultralytics import YOLO


def run_inference():
    model_path = 'weights/best.pt'
    sample_image_path = '../assets/streak_virus.jpg'

    if not os.path.exists(model_path):
        print(f"[Error] Model weights not found at: {model_path}")
        print("Please ensure you have trained the model or downloaded the pre-trained weights.")
        return

    if not os.path.exists(sample_image_path):
        print(f"[Error] Sample image not found at: {sample_image_path}")
        return

    model = YOLO(model_path)

    results = model.predict(source=sample_image_path, conf=0.30, save=True, show=False)
    print("\n--- Detection Results ---")
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("No diseases detected in this image.")
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Print detected class and confidence score
            print(f"- Found: {class_name} | Confidence: {confidence:.2f}")

    print("-------------------------\n")


if __name__ == '__main__':
    run_inference()