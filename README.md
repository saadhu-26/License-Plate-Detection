# License-Plate-Detection
## Aim
To detect a vehicle license plate from an image using OpenCV Haar Cascade Classifier and blur the detected license plate region to protect the vehicle owner's privacy.

## Software Required
Python 3.x

Jupyter Notebook / JupyterLab

OpenCV (cv2)

Matplotlib

Haar Cascade license plate classifier

Input image: car_plate.jpg

Cascade file: haarcascade_licence_plate_rus_16stages.xml

Install Required Libraries

pip install opencv-python numpy matplotlib

## Algorithm
Import OpenCV, NumPy, and Matplotlib.

Read the input vehicle image using cv2.imread().

Load the license plate Haar Cascade XML classifier.

Use detectMultiScale() to detect possible license plate regions.

For every detected plate, obtain its coordinates (x, y, w, h).

Draw a rectangle around the detected license plate to verify detection.

Extract the detected license plate as a Region of Interest (ROI).

Apply median blur to the ROI using cv2.medianBlur().

Replace the original license plate region with the blurred ROI.

Display the final image.

## Program
### Developed By : SAADHANA A
### Reg No: 212225240126

```
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
# Read image
img = cv2.imread('car_plate.png')
# Display image
def display(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
display(img)
# Load license plate Haar Cascade
plate_cascade = cv2.CascadeClassifier(
    'haarcascade_russian_plate_number.xml'
)
# Detect license plate and draw rectangle
def detect_plate(img):
    img_copy = img.copy()
    
    plates = plate_cascade.detectMultiScale(
        img_copy,
        scaleFactor=1.05,
        minNeighbors=3
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(
            img_copy,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    return img_copy

# Display detected plate
result = detect_plate(img)
display(result)
# Detect and blur license plate
def detect_and_blur_plate(img):
    img_copy = img.copy()

    plates = plate_cascade.detectMultiScale(
        img_copy,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]

        # Apply median blur
        blurred_roi = cv2.medianBlur(roi, 25)

        # Replace original ROI with blurred ROI
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

# Display final result
result = detect_and_blur_plate(img)
display(result)
```

## Output

<img width="731" height="489" alt="Screenshot 2026-09-20 162542" src="https://github.com/user-attachments/assets/c6336ef6-bc8d-44c7-a68d-fcbfeb08e10f" />

<img width="720" height="480" alt="Screenshot 2026-09-20 162604" src="https://github.com/user-attachments/assets/9191a96f-7884-4322-a1bb-9a3a133acce6" />

<img width="726" height="483" alt="Screenshot 2026-09-20 162627" src="https://github.com/user-attachments/assets/9740926c-9e33-4e84-aadb-5ae556a76ca9" />

## Result
Thus, the vehicle license plate was successfully detected using the Haar Cascade Classifier in OpenCV and the detected license plate region was successfully blurred using a median blur filter.



