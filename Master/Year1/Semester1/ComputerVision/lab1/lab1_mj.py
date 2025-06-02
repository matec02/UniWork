import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np
import random

def recover_affine_diamond(Hs,Ws,Hd,Wd):
    '''
    A:
    [x1, y1, 0,  0,  1,  0],
    [0,  0,  x1, y1, 0,  1],
    [x2, y2, 0,  0,  1,  0],
    [0,  0,  x2, y2, 0,  1],
    [x3, y3, 0,  0,  1,  0],
    [0,  0,  x3, y3, 0,  1]

    b:
    [x1'],
    [y1'],
    [x2'],
    [y2'],
    [x3'],
    [y3']
    '''

    M = np.array([
        [Ws//2, 0, 0,  0,  1,  0],
        [0,  0,  Ws//2, 0, 0,  1],
        [Ws, Hs//2, 0,  0,  1,  0],
        [0,  0,  Ws, Hs/2, 0,  1],
        [Ws//2, Hs, 0,  0,  1,  0],
        [0,  0,  Ws // 2, Hs, 0,  1]
    ])
    dest_point = np.array([
        [0],
        [0],
        [Wd],
        [0],
        [Wd],
        [Hd]
    ])
    solution = np.linalg.solve(M, dest_point)
    A = np.array([[solution[0][0], solution[1][0]],
                  [solution[2][0], solution[3][0]]])

    b = np.array([solution[4][0], solution[5][0]])
    return A, b

def rmse(i1, i2):
    mse = np.mean((i1 - i2) ** 2)
    return np.sqrt(mse)


def affine_nn(Is, A, b, Hd, Wd):
    output_image = np.zeros((Hd, Wd, Is.shape[2])) if Is.ndim == 3 else np.zeros((Hd, Wd))
    A_inv = np.linalg.inv(A)

    for i in range(Hd):
        for j in range(Wd):
            razlika = np.array([j, i]) - b
            source_coords = np.dot(A_inv, razlika)
            source_x, source_y = np.round(source_coords).astype(int)

            if (0 <= source_x < Is.shape[1]) and (0 <= source_y < Is.shape[0]):
                output_image[i, j] = Is[source_y, source_x]

    return output_image


def affine_bilin(Is, A, b, Hd, Wd):

    output_image = np.zeros((Hd, Wd, Is.shape[2])) if Is.ndim == 3 else np.zeros((Hd, Wd))

    A_inv = np.linalg.inv(A)

    for i in range(Hd):
        for j in range(Wd):
            razlika = np.array([j, i]) - b
            source_coords = np.dot(A_inv, razlika)
            source_x, source_y = source_coords

            if (0 <= source_x < Is.shape[1]-1) and (0 <= source_y < Is.shape[0]-1):
                x0, y0 = np.floor([source_x, source_y]).astype(int)
                dx, dy = source_x - x0, source_y - y0

                if Is.ndim == 3:
                    for c in range(Is.shape[2]):
                        output_image[i, j, c] = (
                                Is[y0, x0, c] * (1 - dx) * (1 - dy) +
                                Is[y0, x0+1, c] * dx * (1 - dy) +
                                Is[y0+1, x0, c] * (1 - dx) * dy +
                                Is[y0+1, x0+1, c] * dx * dy
                        )
                else:
                    output_image[i, j] = (
                            Is[y0, x0] * (1 - dx) * (1 - dy) +
                            Is[y0, x0+1] * dx * (1 - dy) +
                            Is[y0+1, x0] * (1 - dx) * dy +
                            Is[y0+1, x0+1] * dx * dy
                    )
    return output_image


def recover_projective(Qs, Qd):
    num_points = Qs.shape[0]
    A=[]
    for i in range(num_points):
        x_s, y_s = Qs[i][0], Qs[i][1]
        x_d, y_d = Qd[i][0], Qd[i][1]

        array1 = [-x_s, -y_s, -1, 0, 0, 0, x_s * x_d, y_s * x_d, x_d]
        array2 = [0, 0, 0, -x_s, -y_s, -1, x_s * y_d, y_s * y_d, y_d]

        A.append(array1)
        A.append(array2)

    A = np.array(A)
    _,_, solution = np.linalg.svd(A)

    h = solution[-1, :]
    H = h.reshape((3, 3))

    return H

def projective_nn(Is, H, Hd, Wd):

    output_image = np.zeros((Hd, Wd, Is.shape[2])) if Is.ndim == 3 else np.zeros((Hd, Wd))
    H_inv = np.linalg.inv(H)

    for i in range(Hd):
        for j in range(Wd):
            point_d_homog = np.array([j,i,1])
            source_coords = np.dot(H_inv, point_d_homog)

            x, y = source_coords[0] / source_coords[2], source_coords[1] / source_coords[2]

            if (0 <= x < Is.shape[1]) and (0 <= y < Is.shape[0]):
                output_image[i, j] = Is[y.astype(int), x.astype(int)]

    return output_image


def plot_images(images, title="Slike"):
    fig = plt.figure()

    if len(images[0].shape) == 2:
        plt.gray()

    for i, im in enumerate(images):
        fig.add_subplot(1, len(images), i + 1)
        plt.imshow(im.astype(int))

    plt.suptitle(title)
    plt.show()


def main():

    Is = misc.face()
    Is = np.asarray(Is)

    Hd, Wd = 200, 200

    A = 0.25 * np.eye(2) + np.random.normal(size=(2, 2))
    source_center = np.array([Is.shape[1] / 2, Is.shape[0] / 2])
    destination_center = np.array([Wd / 2, Hd / 2])
    b = destination_center - np.dot(A, source_center)

    Id1 = affine_nn(Is, A, b, Hd, Wd)
    Id2 = affine_bilin(Is, A, b, Hd, Wd)

    print(f"Korijen srednjeg kvadratnog odstupanja dvaju slika u zad1 je {rmse(Id1, Id2)}")
    plot_images([Is, Id1, Id2], title="Zad1")

    A, b = recover_affine_diamond(Is.shape[0], Is.shape[1], Hd, Wd)

    Id1 = affine_nn(Is, A, b, Hd, Wd)
    Id2 = affine_bilin(Is, A, b, Hd, Wd)

    print(f"Korijen srednjeg kvadratnog odstupanja dvaju slika u zad2 je {rmse(Id1, Id2)}")
    plot_images([Is, Id1, Id2], title="Zad2")

    #Is = misc.ascent()
    #Is = np.asarray(Is)
    Qd = np.array([
        [0, 0],
        [Wd, 0],
        [Wd, Hd],
        [0, Hd]
    ])
    for i in range(5):
        Qs = []
        for j in range(4):
            x = random.randint(0, Is.shape[1] - 1)
            y = random.randint(0, Is.shape[0] - 1)
            Qs.append([x,y])
        Qs = np.array(Qs)
        H = recover_projective(Qs, Qd)
        Id_projective = projective_nn(Is, H, Hd, Wd)
        plot_images([Is, Id_projective], title=f"Zad3 - iteracija{i+1}")

main()
