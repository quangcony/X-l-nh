import cv2 as cv
import matplotlib.pyplot as plt

def imgThreshold(img, threshold):
    return img > threshold

def showImgThreshold():
    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.subplots(1, 2)

    im1 = cv.imread('./images/moon.png', 0)
    ax1.imshow(im1, cmap = 'gray')
    ax1.set_title('Original image')

    im2 = imgThreshold(im1, 117.0)
    ax2.imshow(im2, cmap = 'gray')
    ax2.set_title('Thresholding image')

    plt.show()

if __name__ == '__main__':
    showImgThreshold()
