import cv2
import numpy as np
import cv2 as cv
img = cv.imread("112.jpg")
cv.imshow('Result', img)
nimg = cv.resize(img, (640, 480))
cv.imshow('Resized', nimg)
gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gimg)
bimg = cv.blur(img, (10, 10))
cv.imshow('Blur', bimg)
gbimg = cv.GaussianBlur(img, (5, 5), 0.0)
cv.imshow('Gauss', gbimg)
blimg = cv.bilateralFilter(img, 9, 75, 75)
cv.imshow('Bilateral', blimg)
brdimg = cv.copyMakeBorder(img, 3, 1, 3, 1, cv.BORDER_CONSTANT)
print(img.shape)
cv.waitKey(0)

cv.destroyAllWindows()

rotateimg = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imshow('Rotate', rotateimg)
blendimg = cv.addWeighted(blimg, 0.5, img, 0.5, 0.5)
cv.imshow('Blend', blendimg)
print(blimg.shape[0], blimg.shape[1])
cv.waitKey(0)
