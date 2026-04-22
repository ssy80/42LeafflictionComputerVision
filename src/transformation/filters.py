import cv2
import numpy as np
from plantcv import plantcv as pcv


def gaussian_blur(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)

    blur_img = pcv.threshold.otsu(blur, object_type='light')

    return blur_img


def mask(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    Apply mask to image
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask_ = pcv.threshold.otsu(blur, object_type='light')

    mask_ = pcv.fill(mask_, size=50)
    mask_ = pcv.median_blur(mask_, ksize=3)

    masked_img = pcv.apply_mask(
        img=img,
        mask=mask_,
        mask_color='white'
    )

    return masked_img


def roi(img: np.ndarray):
    """
    Get HSV from BGR
    Extract saturation (s) channel
    Define green range
    Get healthy mask from green range
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    Get final healthy mask from matching threshold mask
    Overlay final healthy mask to the img
    Draw blue rectangle
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    lower_green = (25, 25, 30)
    upper_green = (95, 255, 255)

    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask_ = pcv.threshold.otsu(blur, object_type='light')
    mask_ = pcv.fill(mask_, size=50)
    mask_ = pcv.median_blur(mask_, ksize=3)

    final_healthy_mask = cv2.bitwise_and(healthy_mask, mask_)

    overlay = img.copy()
    overlay[final_healthy_mask > 0] = (0, 255, 0)

    h, w = overlay.shape[:2]

    cv2.rectangle(
        overlay,
        (0, 0),
        (w-1, h-1),
        (255, 0, 0),
        3
    )

    return overlay


def analyze(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    use plantcv analyze
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask_ = pcv.threshold.otsu(blur, object_type='light')
    mask_ = pcv.fill(mask_, size=50)
    mask_ = pcv.median_blur(mask_, ksize=3)

    pcv.params.text_size = 0
    pcv.outputs.clear()

    analyze_img = pcv.analyze.size(
        img=img,
        labeled_mask=mask_,
        n_labels=1,
        label="leaf_data"
        )

    return analyze_img


def pseudolandmarks(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    Use pseudolandmarks y-axis and draw landmarks on image
    Return the image created
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask_ = pcv.threshold.otsu(blur, object_type='light')
    mask_ = pcv.fill(mask_, size=50)
    mask_ = pcv.median_blur(mask_, ksize=3)

    top_x, bottom_x, center_v = pcv.homology.y_axis_pseudolandmarks(
        img=img, mask=mask_
    )

    result = img.copy()
    for group, color in [(top_x, (255, 0, 0)), (bottom_x, (0, 255, 0)), (center_v, (0, 0, 255))]:
        for point in group:
            x, y = int(point[0][0]), int(point[0][1])
            cv2.circle(result, (x, y), 5, color, -1)

    return result
