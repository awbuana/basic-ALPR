import cv2
import numpy as np

img = cv2.imread('angka-3.jpg', 0)
img2 = img.copy()

templates = []
angka0 = cv2.imread('angka/0.jpg', 0)
angka0 = cv2.resize(angka0,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka0)
angka1 = cv2.imread('angka/1.jpg', 0)
angka1 = cv2.resize(angka1,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka1)
angka2 = cv2.imread('angka/2.jpg', 0)
angka2 = cv2.resize(angka2,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka2)
angka3 = cv2.imread('angka/3.jpg', 0)
angka3 = cv2.resize(angka3,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka3)
angka4 = cv2.imread('angka/4.jpg', 0)
angka4 = cv2.resize(angka4,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka4)
angka5 = cv2.imread('angka/5.jpg', 0)
angka5 = cv2.resize(angka5,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka5)
angka6 = cv2.imread('angka/6.jpg', 0)
angka6 = cv2.resize(angka6,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka6)
angka7 = cv2.imread('angka/7.jpg', 0)
angka7 = cv2.resize(angka7,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka7)
angka8 = cv2.imread('angka/8.jpg', 0)
angka8 = cv2.resize(angka8,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka8)
angka9 = cv2.imread('angka/9.jpg', 0)
angka9 = cv2.resize(angka9,(28,28), interpolation = cv2.INTER_AREA)
templates.append(angka9)

angka = -1
angka_val = float("-inf") 

ii = 0
for template in templates:
    method = eval('cv2.TM_CCOEFF')
    img = img2.copy()
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > angka_val:
        angka_val = max_val
        angka = ii
    ii+=1

pred =  "["+str(angka)+"]"