import cv2 as cv
import matplotlib.pyplot as plt

def logTransform(c, img):
    return float(c) * cv.log(1.0 + img)

def showLogTransform():
    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.subplots(1, 2)

    im1 = cv.imread('./images/log.tif', 0)
    ax1.imshow(im1, cmap = 'gray')
    ax1.set_title('Original Image')

    im2 = logTransform(2.0, im1)
    ax2.imshow(im2, cmap='gray')
    ax2.set_title('Logarithmic Transforms')

    plt.show()

if __name__ == '__main__':
    showLogTransform()