# TODO: Color balance slider (Brightness, Contrast, Gamma Correction, Auto Brightness)
# TODO: Show histogram of image
# TODO: Median filter
# TODO: Mean filter
# TODO: Gaussian filter

import os
import cv2 as cv
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.utils import shuffle


class Step:
    def __init__(self): ...

    def apply(self): ...


class Brightness(Step):
    def __init__(self, alpha):
        self.alpha = alpha

    def apply(self, img: np.ndarray):
        return self.alpha * img


class Contrast(Step):
    def __init__(self, beta):
        self.beta = beta

    def apply(self, img: np.ndarray):
        return self.beta + img


class GammaCorrection(Step):
    def __init__(self, gamma):
        self.gamma = gamma

    def apply(self, img: np.ndarray):
        return np.power(img, (1 / self.gamma))


class AutoBrightness(Step):
    def __init__(self): ...

    def apply(self, img: np.ndarray):
        # White patch
        # Extract channels
        _red = img[:, :, 2]
        _green = img[:, :, 1]
        _blue = img[:, :, 0]
        # Get max value of each channel
        _max_red = np.max(_red)
        _max_green = np.max(_green)
        _max_blue = np.max(_blue)
        # Get average of max values
        _max = (_max_red + _max_green + _max_blue) / 3
        # Calculate alpha
        _alpha = 255 / _max
        # Apply brightness
        return _alpha * img


def filter2D(img: np.ndarray, kernel: np.ndarray, filter_method: Callable):
    # Padding size to handle border pixels
    padding = kernel.shape[0] // 2
    # Pad the image
    padded_image = cv.copyMakeBorder(
        img, padding, padding, padding, padding, cv.BORDER_REPLICATE)
    # Create an empty output image
    output = np.zeros(img.shape, dtype=np.uint8)
    # Iterate over each pixel in the image
    for i in range(padding, padded_image.shape[0] - padding):
        for j in range(padding, padded_image.shape[1] - padding):
            # Extract the neighborhood of the pixel
            neighborhood = padded_image[i - padding:i +
                                        padding + 1, j - padding:j + padding + 1]
            # Apply median operation
            median_value = filter_method(neighborhood)
            # Assign the median value to the corresponding output pixel
            output[i - padding, j - padding] = median_value
    return output


class MedianFilter(Step):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
        self.kernel = self.__get_kernel()

    def __get_kernel(self):
        return np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)

    def apply(self, img: np.ndarray):
        # return filter2D(img, self.kernel, np.median)
        return cv.medianBlur(img, self.kernel_size)


class MeanFilter(Step):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
        self.kernel = self.__get_kernel()

    def __get_kernel(self):
        return np.ones((self.kernel_size, self.kernel_size), dtype=np.float32) / (self.kernel_size ** 2)

    def apply(self, img: np.ndarray):
        # return filter2D(img, self.kernel, np.mean)
        return cv.blur(img, (self.kernel_size, self.kernel_size))


class GaussianFilter(Step):
    def __init__(self, kernel_size: int = 3, sigma: float = 1):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.__get_kernel()

    def __get_kernel(self):
        # Calculate the kernel radius
        radius = self.kernel_size // 2
        # Create an empty kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        # Calculate the constant factor
        factor = 1 / (2 * np.pi * self.sigma ** 2)
        # Calculate the sum of values for normalization
        total = 0
        # Iterate over each element in the kernel
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # Calculate the Gaussian value
                value = factor * \
                    np.exp(-(i ** 2 + j ** 2) / (2 * self.sigma ** 2))
                # Assign the value to the corresponding element in the kernel
                kernel[i + radius, j + radius] = value
                # Accumulate the value for normalization
                total += value
        # Normalize the kernel
        kernel /= total
        return kernel

    def apply(self, img: np.ndarray):
        # return filter2D(img, self.kernel, np.mean)
        return cv.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)


class EdgeExtractor(Step):
    def __init__(self):
        self.kernel = self.__get_kernel()

    def __get_kernel(self):
        return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    def apply(self, img: np.ndarray):
        # _img = cv.filter2D(img, cv.CV_16S, self.kernel)
        # return np.abs(_img).astype(np.uint8)
        return cv.Laplacian(img, cv.CV_8U)


class HistogramEqualization(Step):
    def __init__(self): ...

    def apply(self, img: np.ndarray):
        _hist, be = np.histogram(img, bins=256, range=(0, 255))
        _hist = _hist.astype(np.float32) / sum(_hist)
        _cdf = np.cumsum(_hist)
        return np.interp(img, be, np.hstack((np.zeros((1)), _cdf)))


class HarrisCornerDetector(Step):
    def __init__(self, threshold: int, block_size: int = 2, ksize: int = 3, k: float = 0.004):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.threshold = threshold

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _img_gray = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
        _img_harris = cv.cornerHarris(
            _img_gray, self.block_size, self.ksize, self.k)
        _img_harris = cv.dilate(_img_harris, None)
        _img[_img_harris > self.threshold * _img_harris.max()] = (0, 255, 255)
        return cv.cvtColor(_img, cv.COLOR_RGB2RGBA)


class CannyEdgeDetector(Step):
    def __init__(self, threshold1: int, threshold2: int):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _img_gray = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
        _img_canny = cv.Canny(_img_gray, self.threshold1, self.threshold2)
        _img[_img_canny > 0] = (0, 255, 255)
        return cv.cvtColor(_img, cv.COLOR_RGB2RGBA)


class HoughLineDetector(Step):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _img_gray = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
        _img_canny = cv.Canny(_img_gray, 50, 200)
        _lines = cv.HoughLines(_img_canny, 1, np.pi / 180, self.threshold)
        if _lines is not None:
            for _line in _lines:
                rho, theta = _line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                _x1 = int(x0 + 1000 * (-b))
                _y1 = int(y0 + 1000 * (a))
                _x2 = int(x0 - 1000 * (-b))
                _y2 = int(y0 - 1000 * (a))
                cv.line(_img, (_x1, _y1), (_x2, _y2), (0, 255, 255), 2)
        return cv.cvtColor(_img, cv.COLOR_RGB2RGBA)


class SnakeSegmentation(Step):
    def __init__(self, topleft: tuple[int, int], bottomright: tuple[int, int], alpha: float = 0.015, beta: float = 10, gamma: float = 0.001):
        self.topleft = topleft
        self.bottomright = bottomright
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.density = 100

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _gray_img = rgb2gray(_img)
        # Corners
        a = (self.topleft[1], self.topleft[0])
        b = (self.topleft[0], self.bottomright[1])
        c = (self.bottomright[1], self.bottomright[0])
        d = (self.bottomright[0], self.topleft[1])
        # Edge points
        top_edge = np.linspace(a, b, num=self.density)
        right_edge = np.linspace(b, c, num=self.density)
        bottom_edge = np.linspace(c, d, num=self.density)
        left_edge = np.linspace(d, a, num=self.density)
        # Initial snake
        snake_init = np.concatenate(
            (top_edge, right_edge, bottom_edge, left_edge))
        # Snake
        _snake = active_contour(gaussian(
            _gray_img, 3, preserve_range=False), snake_init, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        # Draw snake
        cv.drawContours(
            _img, [np.flip(_snake, axis=1).astype(int)], -1, (0, 255, 255), 2)
        # Return
        return cv.cvtColor(_img, cv.COLOR_RGB2RGBA)


class WatershedSegmentation(Step):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _gray = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
        _ret, _thresh = cv.threshold(
            _gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # Noise removal
        _kernel = np.ones((3, 3), np.uint8)
        _opening = cv.morphologyEx(
            _thresh, cv.MORPH_OPEN, _kernel, iterations=2)
        # Sure background area
        _sure_bg = cv.dilate(_opening, _kernel, iterations=3)
        # Finding sure foreground area
        _dist_transform = cv.distanceTransform(_opening, cv.DIST_L2, 5)
        _ret, _sure_fg = cv.threshold(
            _dist_transform, self.threshold * _dist_transform.max(), 255, 0)
        # Finding unknown region
        _sure_fg = np.uint8(_sure_fg)
        _unknown = cv.subtract(_sure_bg, _sure_fg)
        # Marker labelling
        _ret, _markers = cv.connectedComponents(_sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        _markers = _markers + 1
        # Now, mark the region of unknown with zero
        _markers[_unknown == 255] = 0
        _markers = cv.watershed(_img, _markers)
        _img[_markers == -1] = [0, 255, 255]
        return cv.cvtColor(_img, cv.COLOR_RGB2RGBA)


class KMeansSegmentation(Step):
    def __init__(self, k):
        self.k = k

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _vectorize = _img.reshape((-1, 3)).astype(np.float32)
        _kmeans = KMeans(n_clusters=self.k, init='k-means++',
                         max_iter=10, n_init=1).fit(_vectorize)
        _labels = _kmeans.predict(_vectorize)
        _res = _kmeans.cluster_centers_.astype(
            np.uint8)[_labels].reshape(_img.shape)
        return cv.cvtColor(_res, cv.COLOR_RGB2RGBA)


class MeanShiftSegmentation(Step):
    def __init__(self): ...

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _vectorize = _img.reshape((-1, 3)).astype(np.float32)
        _img_sample = shuffle(_vectorize, random_state=0)[:1000]
        _bandwidth = estimate_bandwidth(
            _img_sample, quantile=0.2, n_samples=500)
        _meanshift = MeanShift(bandwidth=_bandwidth,
                               bin_seeding=True).fit(_vectorize)
        _labels = _meanshift.predict(_vectorize)
        _res = _meanshift.cluster_centers_.astype(
            np.uint8)[_labels].reshape(_img.shape)
        return cv.cvtColor(_res, cv.COLOR_RGB2RGBA)


class FaceDetection(Step):
    def __init__(self):
        self.__detector = cv.CascadeClassifier(
            "assets/haarcascade_frontalface_alt2.xml")

    def apply(self, img: np.ndarray):
        _img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        _gray = cv.cvtColor(_img, cv.COLOR_RGB2GRAY)
        _faces = self.__detector.detectMultiScale(_gray, 1.05, 5)
        for (_x, _y, _w, _h) in _faces:
            cv.rectangle(_img, (_x, _y), (_x + _w, _y + _h), (0, 255, 255), 2)
        return cv.cvtColor(_img, cv.COLOR_RGB2RGBA)


class Filters:
    def __init__(self):
        self.steps: list[Step] = []

    def add_step(self, step: Step):
        self.steps.append(step)

    def apply_kernel_filter(self, img: np.ndarray):
        for step in self.steps:
            img = step.apply(img)
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img


def histogram(img: np.ndarray, title: str = 'Histogram'):
    _img = img.copy()
    # Extract channels
    _red = _img[:, :, 2]
    _green = _img[:, :, 1]
    _blue = _img[:, :, 0]
    # Get histogram
    plt.style.use('dark_background')
    plt.hist(_red.ravel(), bins=256, range=(0, 255), color='red', alpha=0.5)
    plt.hist(_green.ravel(), bins=256, range=(
        0, 255), color='green', alpha=0.5)
    plt.hist(_blue.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.5)
    plt.title(title)
    plt.show()


def histogram_hog(img: np.ndarray, bins: int = 9, title: str = 'Histogram of Oriented Gradients'):
    # Change color
    _img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    # Calculate HOG
    hog = cv.HOGDescriptor()
    descriptors = hog.compute(_img)
    # Plot histogram
    plt.style.use('dark_background')
    plt.hist(descriptors, bins=bins)
    plt.title(title)
    plt.show()
