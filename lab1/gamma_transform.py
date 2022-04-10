import cv2 as cv
import matplotlib.pyplot as plt

def gammaTransform(c, img, gamma):
    return float(c) * pow(img, gamma)

def showGammaTransform():
    fig = plt.figure(figsize=(12, 6))
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

    im1 = cv.imread('./images/sanbay.tif', 0)
    ax1.imshow(im1, cmap = 'gray')
    ax1.set_title('Original Image')

    im2 = gammaTransform(1.0, im1, 3.0)
    ax2.imshow(im2, cmap='gray')
    ax2.set_title('Gamma = 3')

    im3 = gammaTransform(1.0, im1, 4.0)
    ax3.imshow(im3, cmap='gray')
    ax3.set_title('Gamma = 4')

    im4 = gammaTransform(1.0, im1, 5.0)
    ax4.imshow(im4, cmap='gray')
    ax4.set_title('Gamma = 5')

    plt.show()

if __name__ == '__main__':
    showGammaTransform()