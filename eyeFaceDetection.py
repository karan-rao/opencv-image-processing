import cv2

# Import the original image
img = cv2.imread("faces.jpeg",1)
# Covert it to grey scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the Machine Learnt classifier paths (Haar Cascade)
path = "haarcascade_eye.xml"
path2 = "haarcascade_frontalface_default.xml"

# Create Cascade classifier objects
eye_cascade = cv2.CascadeClassifier(path)
face_cascade = cv2.CascadeClassifier(path2)

# Detect features at desired tolerance
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=20, minSize=(15, 15))
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40, 40))

# Draw circle for every eye detected
for (x, y, w, h) in eyes:
    xc = (x + x+w)/2
    yc = (y + y+h)/2
    radius = w/2
    cv2.circle(img, (int(xc), int(yc)), int(radius), (255, 0, 0), 2)

# Draw rectangle for every face detected
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 120), 3)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()