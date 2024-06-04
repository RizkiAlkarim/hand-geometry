import numpy as np
import matplotlib.pylab as plt

def convert_to_grayscale(rgb_weight, image):
    image = np.dot(image[..., :3], rgb_weight)
    return image


def main():
    image = plt.imread("./img/hand_1.jpg")
    rgb_weight = [0.21, 0.72, 0.07]
    image = convert_to_grayscale(rgb_weight, image)

    plt.axis("off")
    plt.imshow(image)
    plt.show()

main()
