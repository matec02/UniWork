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


def canny_edge_detector(img, sigma, min_val, max_val):

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

    gradient_abs = np.sqrt(I_x2 + I_y2)
    gradient_orientation = np.arctan2(I_y, I_x)

    max_g_abs = np.max(gradient_abs)
    gradient_abs_normalized = (gradient_abs / max_g_abs) * 255

    plt.imshow(gradient_abs_normalized, cmap="gray")
    plt.title("Normalizirani gradijent")
    plt.show()

    # Zadatak 5
    suppressed = np.zeros_like(gradient_abs)
    theta = gradient_orientation * (180.0 / np.pi)
    theta[theta < 0] += 180.0

    for i in range(1, gradient_abs.shape[0] - 1):
        for j in range(1, gradient_abs.shape[1] - 1):
            q = 255
            r = 255
            val = theta[i, j]

            if (0 <= val < 22.5) or (157.5 <= val <= 180):
                q = gradient_abs[i, j + 1]
                r = gradient_abs[i, j - 1]
            elif 22.5 <= val < 67.5:
                q = gradient_abs[i + 1, j - 1]
                r = gradient_abs[i - 1, j + 1]
            elif 67.5 <= val < 112.5:
                q = gradient_abs[i + 1, j]
                r = gradient_abs[i - 1, j]
            else:
                q = gradient_abs[i - 1, j - 1]
                r = gradient_abs[i + 1, j + 1]

            if (gradient_abs[i, j] >= q) and (gradient_abs[i, j] >= r):
                suppressed[i, j] = gradient_abs[i, j]
            else:
                suppressed[i, j] = 0

    plt.imshow(suppressed, cmap="gray")
    plt.title("Normalizirani gradijent s potiskivanjem")
    plt.show()

    # Zadatak 6
    strong_edges = suppressed >= max_val
    weak_edges = (suppressed >= min_val) & (suppressed < max_val)

    for i in range(1, suppressed.shape[0]-1):
        for j in range(1, suppressed.shape[1]-1):

            if weak_edges[i, j]:
                if (strong_edges[i-1:i+2, j-1:j+2]).any():
                    strong_edges[i, j] = True
                else:
                    suppressed[i, j] = 0

    suppressed[~strong_edges] = 0
    plt.imshow(suppressed, cmap="gray")
    plt.title("Normalizirani gradijent s histerezom")
    plt.show()



def main():
    img_before = np.array(Image.open("../house.jpg"))
    img = img_info_and_return(img_before)
    canny_edge_detector(img=img, sigma=1.5, min_val=10, max_val=90)

main()