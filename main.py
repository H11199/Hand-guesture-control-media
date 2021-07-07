import cv2
import numpy as np
import math
import pyautogui as p
import time

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while (1):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (100,100), (400, 400), (0,255,0), 0)
    crop_image = frame[100:400, 100:400]


    # smoothing image
    blur = cv2.GaussianBlur(crop_image, (5,5), 100)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, np.array([0,20,70]), np.array([20, 255, 255]))
    kernel = np.ones((3,3))

    dilation = cv2.dilate(mask, kernel, iterations=4)
    erosion = cv2.erode(dilation, kernel, iterations=4)

    filtered = cv2.GaussianBlur(erosion, (5,5), 100)
    ret, th = cv2.threshold(filtered, 127, 255, 0)



    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(crop_image.shape, np.uint8)
    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x,y), (x+w, y+h), (0, 0, 255), 0)

        ## find convex hull
        hull = cv2.convexHull(contour)

        ## Draw contour

        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # defects array look like this:
        # np.array([[[[1,2],[3,4],[5,6],10]], [[[1,3],[4,6],[2,4],7]]])
        # shape = (2,1,4) shape[0] = 2
        # Here each row contains hull starting point, ending point, farthest point and distance
        # s,e,f,d = d[0,0]
        # s = [1,2], e = [3,4], f= [5,6], d=10

        count_defects = 0


        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b**2+c**2-a**2)/(2*b*c))*180)/3.14

            if angle<=90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [255,255,255], -1)

            cv2.line(crop_image, start, end, [0,255,0],2)


        if count_defects == 0:
            cv2.putText(frame, "ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)



        elif count_defects == 1:
            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            p.press("volumeup")


        elif count_defects == 2:

            cv2.putText(frame, "THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            p.press("volumedown")



        elif count_defects == 3:
            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            p.press("right")

        elif count_defects == 4:
            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            p.press('space')
        else:
            pass

    except:
        pass

    # Show images
    cv2.imshow("Threshold", th)
    cv2.imshow("Gesture", frame)


    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
