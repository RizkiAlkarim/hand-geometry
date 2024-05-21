import numpy as np
import matplotlib.pylab as plt

def convert_to_grayscale(rgb_weight, image):
    image = np.dot(image[..., :3], rgb_weight)
    return image

def gaussian_filter(image):


def main():
    image = plt.imread("./img/hand_1.jpg")
    rgb_weight = [0.2989, 0.5870, 0.1140]
    image = convert_to_grayscale(rgb_weight, image)

    plt.axis("off")
    plt.imshow(image)
    plt.show()

main()
