import cv2
import torch

# Load the YOLOv9 model
model = torch.hub.load('your-repo/yolov9', 'yolov9', pretrained=True)

# Load image
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Get predictions
detections = results.xyxy[0]  # Bounding boxes

# Count objects
object_count = len(detections)
print(f'Number of detected objects: {object_count}')

# Draw bounding boxes on the image
for *box, conf, cls in detections:
    label = f'{model.names[int(cls)]}: {conf:.2f}'
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.putText(image, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the results
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
