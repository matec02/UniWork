from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def img_info_and_return(img):
    # Zadatak 1
    print(f"Dimenzija slike je {img.shape}")
    if len(img.shape) == 3:
        print("Slika je visebojna")
        img = np.mean(img, axis=2).astype(np.uint8)

    min_img = np.min(img)
    max_img = np.max(img)
    print(f"Minimalna vrijednost je {min_img}, a maksimalna vrijednost je {max_img}")
    print(f"Intenziteti gornjeg lijevog isjecka:\n {img[:10][:10]}")
    print(f"Tip podataka slike je {img.dtype}")
    img = img.astype("float")
    print(f"Tip podataka slike je {img.dtype}")
    return img


def harrison_edge_detector(img, sigma, threshold, k, top_k, kernel_sum, kernel_max_suppresion):

    # Zadatak 2
    img_gauss = gaussian_filter(img, sigma=sigma)
    plt.imshow(img_gauss, cmap="gray")
    plt.title("Gaussovo zagladivanje")
    plt.show()

    # Zadatak 3
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    I_x = convolve(img_gauss, sobel_x)
    I_y = convolve(img_gauss, sobel_y)

    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(I_x, cmap="gray")
    axes[0].set_title(f"I_x")
    axes[1].imshow(I_y, cmap="gray")
    axes[1].set_title(f"I_y")
    plt.show()

    # Zadatak 4
    kernel = np.ones((kernel_sum,kernel_sum))
    S_x2 = convolve(I_x2, kernel)
    S_y2 = convolve(I_y2, kernel)
    S_xy = convolve(I_xy, kernel)

    # Zadatak 5
    det = (S_x2 * S_y2) - S_xy ** 2
    trace = S_x2 + S_y2
    r = det - k*(trace ** 2)
    plt.imshow(r, cmap="gray")
    plt.title("Harrison odziv")
    plt.show()

    # Zadatak 6
    r_threshold = np.where(r > threshold, r, 0)
    r_threshold_max_supression = np.copy(r_threshold)

    offset = kernel_max_suppresion // 2
    for i in range(r_threshold_max_supression.shape[0]):
        for j in range(r_threshold_max_supression.shape[1]):
            top = max(i - offset, 0)
            bottom = min(i + offset + 1, r_threshold_max_supression.shape[0])
            left = max(j - offset, 0)
            right = min(j + offset + 1, r_threshold_max_supression.shape[1])

            window = r_threshold[top:bottom, left:right]
            value = r_threshold[i, j]

            if value < np.max(window):
                r_threshold_max_supression[i, j] = 0

    # Zadatak 7
    nonzero_coordinates = np.nonzero(r_threshold_max_supression)
    nonzero_values_in_image = r_threshold_max_supression[nonzero_coordinates]

    sorted_values = np.argsort(nonzero_values_in_image)[::-1]
    biggest_k = sorted_values[:top_k]
    top_k_coords = (nonzero_coordinates[0][biggest_k], nonzero_coordinates[1][biggest_k])

    plt.imshow(img, cmap='gray')
    plt.scatter(top_k_coords[1], top_k_coords[0], s=30, marker='o', facecolors='none', edgecolors='r')
    plt.title(f"{top_k} najvecih odziva")
    plt.show()



def main():
    print("FER Logo")
    img_fer = np.array(Image.open("../fer_logo.jpg"))
    img_1 = img_info_and_return(img_fer)
    harrison_edge_detector(img=img_1, sigma=1, threshold=1e10, k=0.04, top_k=100,
                           kernel_sum=5, kernel_max_suppresion=14)
    print("\nKuca")
    img_house = np.array(Image.open("../house.jpg"))
    img_2 = img_info_and_return(img_house)
    harrison_edge_detector(img=img_2, sigma=1.5, threshold=1e9, k=0.04, top_k=100,
                           kernel_sum=5, kernel_max_suppresion=32)


main()
