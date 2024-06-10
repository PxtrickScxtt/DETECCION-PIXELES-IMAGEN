# OpenCV Python program to detect cars in video frame
# Import libraries of python OpenCV
import cv2

# Capture frames from a video
cap = cv2.VideoCapture('carrosss.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')

# Check if the cascade classifier was loaded correctly
if car_cascade.empty():
    print("Error: Could not load cascade classifier.")
    exit()

# Loop runs if capturing has been initialized
while True:
    # Read frames from a video
    ret, frames = cap.read()

    # Break the loop if no frame is returned (end of video)
    if not ret:
        break

    # Resize the frame to 640x480
    frames = cv2.resize(frames, (1080, 720))

    # Convert to gray scale of each frame
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw a rectangle around each detected car
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display frames in a window
    cv2.imshow('video2', frames)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cap.release()
cv2.destroyAllWindows()
