import cv2
import numpy as np


def flip(img: np.ndarray):
    """
    Flip image to the right
    """
    return cv2.flip(img, 1)


def rotate(img: np.ndarray):
    """
    Rotate to right 25 degree angle.
    """
    angle = -25
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated


def skew(img: np.ndarray):
    """
    Skew image
    """
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[0, 0], [w, 0], [int(0.2*w), h]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))


def crop(img: np.ndarray):
    """
    Crop image by 10%
    """
    h, w = img.shape[:2]
    y1 = int(0.1 * h)
    y2 = int(0.9 * h)
    x1 = int(0.1 * w)
    x2 = int(0.9 * w)
    return img[y1:y2, x1:x2]


def distortion(img: np.ndarray):
    """
    Blur the image
    """
    return cv2.GaussianBlur(img, (7, 7), 0)
