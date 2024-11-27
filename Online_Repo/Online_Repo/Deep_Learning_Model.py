import cv2
import os
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv5 model (assuming "yolo11n.pt" is a valid model)
model = YOLO("yolo11n.pt")

folder_path = r"C:\Users\cic\Documents\tarazou2\Online_Repo\Image_dataset"
file_names = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

# Iterate over the images in the folder
for file_name in file_names:
    image_path = os.path.join(folder_path, file_name)

    # Run the model on the image
    results = model(image_path)

    # Load the image using cv2
    image = cv2.imread(image_path)

    # Draw bounding boxes and labels on the image
    for result in results:  # Iterate through results (one per detection)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = box.cls[0]  # Class label index
            label_text = f"{model.names[int(label)]} {confidence:.2f}"
            
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the label
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow('YOLO Detections', image)

    # Wait for user input to move to the next image (key press)
    key = cv2.waitKey(0)  # Wait indefinitely for a key press
    if key == 27:  # If ESC key is pressed, exit the loop
        break

cv2.destroyAllWindows()  # Close the OpenCV window
