"""
Генерация выборки изображений для анализа данных.

Будет сгенерировано:
50 grayscale, каждый
"""
import cv2 as cv

main_img = cv.imread('input/main.jpg')
mask_img = cv.imread('input/mask.jpg')

cv.imshow("main", main_img)
cv.imshow("mask", mask_img)

cv.waitKey(0)

cv.imshow("gmain", cv.cvtColor(main_img, cv.COLOR_BGR2GRAY))
cv.imshow("gmask", cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY))

cv.waitKey(0)

print(cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY).shape, mask_img.shape)
