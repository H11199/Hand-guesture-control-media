import cv2
import numpy as np


im = cv2.imread('static/images/handgray.jpg')
print(im.shape)
imgresized = cv2.resize(im,(600,600))
imgray = cv2.cvtColor(imgresized, cv2.COLOR_BGR2GRAY)
cv2.medianBlur(imgray, 5)
rct, th3 = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

## Draw hull for each contour
hull = [cv2.convexHull(c) for c in contours]

final = cv2.drawContours(th3.copy(), hull, -1, (255, 0, 0))


cv2.imshow('Contours', th3)
cv2.imshow('original', imgray)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()