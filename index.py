import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
def load_model(model_path): ''
    return tf.saved_model.load(model_path)

# Object detection on a single image
def detect_objects(image, model):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    return detections

# Draw bounding boxes on the image
def draw_boxes(image, detections, threshold=0.5):
    height, width, _ = image.shape

    for i in range(detections['num_detections']):
        score = detections['detection_scores'][i]
        if score > threshold:
            box = detections['detection_boxes'][i]
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return image

def main():
    # Load the pre-trained object detection model
    model_path = ''
    model = load_model(model_path)

    # Open the video file
    video_path = 'ball.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection on the current frame
        detections = detect_objects(frame, model)

        # Draw bounding boxes
        frame_with_boxes = draw_boxes(frame, detections)

        # Show the frame with bounding boxes
        cv2.imshow('Object Detection', frame_with_boxes)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
