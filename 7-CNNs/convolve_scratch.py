import numpy as np
import math
from PIL import Image

def load_image_channel(path):
    img = Image.open(path)
    rgb = np.array(img.convert('RGB'))
    g = rgb[:,:,1] # green channel only
    Image.fromarray(np.uint8(g)).save("green.png")
    return g

def convolve(image, kernel):
    # make sure kernel is a square martrix
    assert kernel.shape[0] == kernel.shape[1], "Kernel is not square"
    # this represents the pixels around the border that will be lost from convolving
    buffer = math.floor(kernel.shape[1]/2)
    out_width = image.shape[1]-(2*buffer)
    out_height = image.shape[0]-(2*buffer)
    output = np.zeros((out_height, out_width))

    for i, x in enumerate(range(buffer, image.shape[1]-buffer)):
        for j, y in enumerate(range(buffer, image.shape[0]-buffer)):
            # convolve kernel with image 
            # element-wise multipliy and sum       
            output[j, i] = (kernel * image[y-buffer : y+buffer+1, x-buffer : x+buffer+1]).sum()
    return output


if __name__ == "__main__":
    image = load_image_channel("house.png")
    # image = np.ones((20,20))
    kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

    result = convolve(image, kernel1)
    Image.fromarray(np.uint8(result)).save("kernel1.png")
    result = convolve(image, kernel2)
    Image.fromarray(np.uint8(result)).save("kernel2.png")
