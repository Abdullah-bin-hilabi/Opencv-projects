import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#it is like imshow but i can resize the window
cv2.namedWindow("bars")
cv2.resizeWindow("bars", 400, 300)

# this is the color i want to be invisible in my case its dark red you can change it to any color you want
cv2.createTrackbar("u_hue", "bars", 180, 180, lambda x: None)
cv2.createTrackbar("u_sat", "bars", 255, 255, lambda x: None)
cv2.createTrackbar("u_val", "bars", 200, 255, lambda x: None)
cv2.createTrackbar("l_hue", "bars", 170, 180, lambda x: None)
cv2.createTrackbar("l_sat", "bars", 150, 255, lambda x: None)
cv2.createTrackbar("l_val", "bars", 80, 255, lambda x: None)

# Grab the initial frame (cloak background)
ret, initial_frame = cap.read()
if not ret:
    print("Failed to capture initial frame. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read trackbar positions
    u_hue = cv2.getTrackbarPos("u_hue", "bars")
    u_sat = cv2.getTrackbarPos("u_sat", "bars")
    u_val = cv2.getTrackbarPos("u_val", "bars")

    l_hue = cv2.getTrackbarPos("l_hue", "bars")
    l_sat = cv2.getTrackbarPos("l_sat", "bars")
    l_val = cv2.getTrackbarPos("l_val", "bars")

    # Define HSV ranges
    upper_hsv = np.array([u_hue, u_sat, u_val])
    lower_hsv = np.array([l_hue, l_sat, l_val])

    # Create mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 3)
    mask_inv = 255 - mask

    # Dilation kernel
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, 5)

    # Mask the current frame
    b, g, r = cv2.split(frame)
    b = cv2.bitwise_and(b, mask_inv)
    g = cv2.bitwise_and(g, mask_inv)
    r = cv2.bitwise_and(r, mask_inv)
    frame_inv = cv2.merge((b, g, r))

    # Mask the initial frame (cloak background)
    b, g, r = cv2.split(initial_frame)
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    cloak_area = cv2.merge((b, g, r))

    # Combine both
    final = cv2.bitwise_or(frame_inv, cloak_area)

    cv2.imshow("invisible_cloak", final)
    # cv2.imshow("original", frame)

    if cv2.waitKey(3) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
