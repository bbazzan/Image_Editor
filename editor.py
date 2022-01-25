from image import Image
import numpy as np


def adjust_brightness(image, factor):
    """Multiplies the brightness value of each pixel by the factor;
    factor > 1 ==> brightens the image, factor < 1 ==> darkens the image"""

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=y_pixels, y_pixels=y_pixels, num_channels=num_channels)
    new_im.array = image.array * factor

    return new_im


def adjust_contrast(image, factor, mid=0.5):
    """Adjusts the contrast by increasing the difference from the user-defined midpoint by factor amount"""

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)
    new_im.array = (image.array - mid) * factor + mid

    return new_im


def blur(image, kernel_size):
    """Takes a surrounding of a pixel and applies blur. The kernel size is the number of pixels to take into
    account when applying blur; it should always be an odd number"""

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)
    neighbor_range = kernel_size // 2
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, x - neighbor_range), min(new_im.x_pixels - 1, x + neighbor_range) + 1):
                    for y_i in range(max(0, y - neighbor_range), min(new_im.y_pixels - 1, y + neighbor_range) + 1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)

    return new_im


def apply_kernel(image, kernel):
    """Applies a 2D kernel. It assumes that the kernel is a square"""

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)
    neighbor_range = kernel.shape[0] // 2
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, x - neighbor_range), min(new_im.x_pixels - 1, x + neighbor_range) + 1):
                    for y_i in range(max(0, y - neighbor_range), min(new_im.y_pixels - 1, y + neighbor_range) + 1):
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_im.array[x, y, c] = total
    return new_im


def combine_two_images(image1, image2):
    """Combines two images using the squared sum of squares. The size of both images must be the same"""

    x_pixels, y_pixels, num_channels = image1.array.shape
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)
    new_im.array = np.sqrt(image1.array ** 2 + image2.array ** 2)

    return new_im


if __name__ == '__main__':
    city = Image(filename='city.png')

    # brightening
    brightened_im = adjust_brightness(city, 1.8)
    brightened_im.write_image('brightened_city.png')

    # darkening
    darkened_im = adjust_brightness(city, 0.2)
    darkened_im.write_image('darkened_city.png')

    # high contrast
    increased_contrast_im = adjust_contrast(city, 2, 0.5)
    increased_contrast_im.write_image('high_contrast_city.png')

    # low contrast
    decreased_contrast_im = adjust_contrast(city, 0.5, 0.5)
    decreased_contrast_im.write_image('low_contrast_city.png')

    # blur using kernel of 3
    blur_3_im = blur(city, 3)
    blur_3_im.write_image('blur_k3_city.png')

    # blur using kernel of 15
    blur_15_im = blur(city, 15)
    blur_15_im.write_image('blur_k15_city.png')

    # sobel edge detection kernel on the x and y-axis
    sobel_x_im = apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    sobel_x_im.write_image('sobel_x_city.png')
    sobel_y_im = apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y_im.write_image('sobel_y_city.png')

    # combined sobel_x and sobel_y to make edge detector
    sobel_xy_im = combine_two_images(sobel_x_im, sobel_y_im)
    sobel_xy_im.write_image('edge_xy_city.png')
