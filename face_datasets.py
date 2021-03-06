# Import OpenCV2 for image processing
import cv2

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = 1

# Initialize sample face image
count = 0

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

  # use histogram equalizer to enhance  contrast
    equ = cv2.equalizeHist(gray)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(equ, 1.3, 5)

    #if not face print not face
    if (len(faces) == 0):
        print ("No Face") 

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1
        
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h-20,x+20:x+w-10])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # Close the program when Press "ESC"
    k = cv2.waitKey(10)
    if k == 27:
        break

    # If image taken reach 100, stop taking video
    elif count>100:
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
