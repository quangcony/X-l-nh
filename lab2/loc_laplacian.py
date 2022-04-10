import cv2     # Thư viện OpenCV
import numpy as np   # Thư viện numy để làm việc dữ liệu kiểu mảng
import matplotlib.pyplot as plt # import thư viện matplotlib để vẽ ảnh

# Định nghĩa hàm Tich_chap() để lọc Trung bình, Trung bình có trọng số và lọc Gaussian
def Tich_chap(img,mask):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp   =  img[i-1, j-1]    * mask[0, 0]\
                   +  img[i, j-1]      * mask[0, 1]\
                   +  img[i+1, j - 1]  * mask[0, 2]\
                   +  img[i-1, j]      * mask[1, 0]\
                   +  img[i, j]        * mask[1, 1]\
                   +  img[i+1, j]      * mask[1, 2]\
                   +  img[i - 1, j+1]  * mask[2, 0]\
                   +  img[i, j + 1]    * mask[2, 1]\
                   +  img[i + 1, j + 1]* mask[2, 2]
            img_new[i, j]= temp
    img_new = img_new.astype(np.uint8)
    return img_new

# Định nghĩa bộ lọc Laplacian chuẩn
locLaplacian_chuan = np.array(([0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]), dtype="float")

# Định nghĩa bộ lọc Laplacian biến thể 1
locLaplacian_Bien_The = np.array(([1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]), dtype="float")

locLaplacian_Cai_Thien = np.array(([-1, -1, -1],
                                    [-1, 5, -1],
                                    [-1, -1, -1]), dtype="float")

fig = plt.figure(figsize=(16, 9)) # Tạo vùng vẽ tỷ lệ 16:9
(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2, 3) # Tạo 6 vùng vẽ con

# Đọc và hiển thị ảnh gốc
image = cv2.imread('images/loc_laplacian.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Ảnh gốc")

# Lọc Laplacian chuẩn và hiển thị ảnh
img_loc_Laplacian_chuan = Tich_chap(image, locLaplacian_chuan) #Gọi hàm tích chập
ax2.imshow(img_loc_Laplacian_chuan, cmap='gray')
ax2.set_title("Ảnh lọc Laplace chuẩn")

# Tạo ảnh cải thiện từ lọc Laplacian chuẩn và hiển thị ảnh
img_cai_thien_locLaplacian_chuan = image - img_loc_Laplacian_chuan
ax3.imshow(img_cai_thien_locLaplacian_chuan, cmap='gray')
ax3.set_title("Ảnh Cải thiện từ Laplace chuẩn")

# Lọc Laplacian Biến thể thứ nhất và hiển thị ảnh
img_loc_Laplacian_Bien_The = Tich_chap(image, locLaplacian_Bien_The) #Gọi hàm tích chập
ax4.imshow(img_loc_Laplacian_Bien_The, cmap='gray')
ax4.set_title("Ảnh lọc Laplace biến thể")

# Tạo ảnh cải thiện từ lọc Laplacian Biến thể thứ nhất và hiển thị ảnh
img_cai_thien_loc_Laplacian_Bien_The = image - img_loc_Laplacian_Bien_The
ax5.imshow(img_cai_thien_loc_Laplacian_Bien_The, cmap='gray')
ax5.set_title("Ảnh Cải thiện từ Laplace biến thể")

ax6.axis('off')

# Hiển thị vùng vẽ
plt.show()