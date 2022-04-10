import cv2 as cv

# reading image
img = cv.imread('./images/junvu.jpg')

# image dimension
print('img.shape:', img.shape)

# show image
cv.imshow('Jun Vu Model', img)

cv.waitKey(0)