# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # use histogram equalizer to enhance  contrast
    equ = cv2.equalizeHist(gray)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(equ, 1.3,5)

    #if not face print not face
    if (len(faces) == 0):
        cv2.putText(im, str("No Face"), (0,50), font, 1, (255,0,255), 2)
        cv2.putText(im, str("Lock"), (260,450), font, 1, (0,0,255), 2)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        #changed gray with equ
        Id = recognizer.predict(equ[y:y+h,x:x+w])

        # take a presentage of recognize 
        presentage = str(Id[1])[:4]

        if(Id[1] > 90):
            cv2.putText(im, str("UnLock"), (240,450), font, 1, (0,155,0), 2)
        else:
            cv2.putText(im, str("Lock"), (260,450), font, 1, (0,0,255), 2)
        
        # Check the ID if exist
        if( 1 in Id):
            Id = "ALI"
       # elif( 2 in Id):
       #    Id = "A'mer"
    
            
        #If not exist, then it is Unknown
        else:
            Id = "Unknow"
	    
        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)
        cv2.putText(im, str("Match rate : "+presentage+"%"), (150,50), font, 1, (0,155,0), 2)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # Close the program when Press "ESC"
    k = cv2.waitKey(10)
    if k == 27:
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
