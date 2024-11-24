import cv2
import os

# File paths
cascade_path = r'C:\Users\lenovo\Desktop\NIT FILES\21st- python to mysql connection\haar cascade classifier basic project\Haarcascades\haarcascade_frontalface_default.xml'
image_path = r'D:\FAMILY PICS.VIDEOS\pictures of my phn\Ding dong\IMG-20240504-WA0084.jpg'

# Check file paths
if not os.path.exists(cascade_path):
    print(f"Haar Cascade file not found: {cascade_path}")
    exit()
if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
    exit()

# Load Haar cascade
face_classifier = cv2.CascadeClassifier(cascade_path)
if face_classifier.empty():
    print(f"Failed to load Haar cascade: {cascade_path}")
    exit()

# Load and verify image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image: {image_path}")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

if len(faces) == 0:
    print("No faces found")
else:
    print(f"{len(faces)} faces found")
    for (x, y, w, h) in faces:
        print(f"Face detected: Top-left=({x}, {y}), Width={w}, Height={h}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    # Save the output image
    cv2.imwrite(r'C:\Users\lenovo\Desktop\face_detection_output.jpg', image)
    print("Output image saved at: C:\\Users\\lenovo\\Desktop\\face_detection_output.jpg")

    # Resize the image for display
    resized_image = cv2.resize(image, (800, 600))  # Adjust dimensions if needed

    # Display the image
    cv2.imshow('Face Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






