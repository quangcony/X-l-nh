import cv2 as cv
import matplotlib.pyplot as plt

def imgInverse(img):
    return 255 - img

def showImgInverse():
    fig = plt.figure(figsize=(12, 6))
    ax1, ax2 = fig.subplots(1, 2)

    im1 = cv.imread('./images/daoanh.tif', 0)
    ax1.imshow(im1, cmap = 'gray')
    ax1.set_title('Original image')

    im2 = imgInverse(im1)
    ax2.imshow(im2, cmap = 'gray')
    ax2.set_title('Inverse image')

    plt.show()

if __name__ == '__main__':
    showImgInverse()
