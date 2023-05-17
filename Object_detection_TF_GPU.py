import tensorflow as tf
import numpy as np
import cv2

# Enable GPU acceleration
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the pre-trained object detection model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the input image
image = cv2.imread('input_image.jpg')

# Preprocess the image
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (224, 224))
input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

# Add a batch dimension to the input image
input_image = np.expand_dims(input_image, axis=0)

# Perform object detection
with tf.device('/GPU:0'):
    predictions = model.predict(input_image)

# Get the class labels and corresponding confidence scores
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

# Display the top 5 predictions
for label, _, confidence in decoded_predictions:
    print(f"{label}: {confidence*100:.2f}%")

# Draw bounding boxes on the image
class_ids = np.argmax(predictions, axis=-1)
class_label = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0][0][1]
class_confidence = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0][0][2]

if class_confidence > 0.5:
    image_height, image_width, _ = image.shape
    box = predictions[0, 0, 0, :4] * np.array([image_width, image_height, image_width, image_height])
    xmin, ymin, xmax, ymax = box.astype(np.int)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, f"{class_label}: {class_confidence*100:.2f}%", (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the resulting image with bounding box and label
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
