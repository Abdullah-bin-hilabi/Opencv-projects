import cv2

face_cascade = cv2.CascadeClassifier("C:/Users/Hp/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in face:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #we are going to use the median Blur function to blur the face
        img[y:y+h, x:x+w] = cv2.medianBlur(img[y:y+h, x:x+w], 35)
    
    cv2.imshow("blur the face", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()