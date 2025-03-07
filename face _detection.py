import cv2

#it is the face detection code
#i am using the haarcascade_frontalface_default.xml file to detect the face
face_cascade = cv2.CascadeClassifier("C:/Users/Hp/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
#it is for the video capture
video_cap = cv2.VideoCapture(0)
while True:
    ret, video = video_cap.read()
    #i change the color of the video to gray becuse it is easy to detect the face in gray color
    col = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    #face decetection box code
    faces = face_cascade.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    for x,y,w,h in faces:
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video)
    if cv2.waitKey(1) == ord('q'):
        break
video_cap.release()