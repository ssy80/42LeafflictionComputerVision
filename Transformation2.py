import cv2
import sys
import os
import numpy as np
from pathlib import Path
import argparse
from utils import is_path_dir
from plantcv import plantcv as pcv
from skimage.morphology import skeletonize
import re
import glob
import matplotlib.pyplot as plt


def is_image_file(filepath: Path)-> None:
    """
    Check filepath is an image file (.jpg, .jpeg) 
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"file not found: {filepath}")

    if filepath.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("file must be a .jpg or .jpeg image")


'''def draw_pseudolandmarks(img, top, bottom, center_v, radius=4):
    out = img.copy()

    def draw_points(points, color):
        xs = np.asarray(points[0]).ravel()
        ys = np.asarray(points[1]).ravel()

        for x, y in zip(xs, ys):
            cv2.circle(out, (int(float(x)), int(float(y))), radius, color, -1)

    draw_points(top, (0, 255, 0))       # green
    draw_points(bottom, (255, 0, 0))    # blue
    draw_points(center_v, (0, 0, 255))  # red

    return out
'''


def plot_leaf_color_histogram(img, mask=None):
    """
    img: BGR image from cv2.imread(...)
    mask: binary mask, 255 = keep pixel, 0 = ignore
    """
    # ensure 3-channel image
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

    # Keep only masked pixels
    valid = mask > 0

    # Convert color spaces
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extract channels inside mask
    red = rgb[:, :, 0][valid]
    green = rgb[:, :, 1][valid]
    blue = rgb[:, :, 2][valid]

    hue = hsv[:, :, 0][valid]
    saturation = hsv[:, :, 1][valid]
    value = hsv[:, :, 2][valid]

    lightness = lab[:, :, 0][valid]
    green_magenta = lab[:, :, 1][valid]
    blue_yellow = lab[:, :, 2][valid]

    channels = {
        "blue": blue,
        "blue-yellow": blue_yellow,
        "green": green,
        "green-magenta": green_magenta,
        "hue": hue,
        "lightness": lightness,
        "red": red,
        "saturation": saturation,
        "value": value,
    }

    colors = {
        "blue": "blue",
        "blue-yellow": "yellow",
        "green": "green",
        "green-magenta": "magenta",
        "hue": "#7d3cff",
        "lightness": "gray",
        "red": "red",
        "saturation": "cyan",
        "value": "orange",
    }

    plt.figure(figsize=(11, 6))

    total_pixels = np.count_nonzero(valid)

    for name, vals in channels.items():
        hist, bins = np.histogram(vals, bins=256, range=(0, 256))
        hist_percent = (hist / total_pixels) * 100
        plt.plot(bins[:-1], hist_percent, label=name, color=colors[name], linewidth=1.5)

    plt.xlabel("Pixel intensity", fontsize=16)
    plt.ylabel("Proportion of pixels (%)", fontsize=16)
    plt.xlim(0, 255)
    plt.legend(title="color Channel", bbox_to_anchor=(1.03, 0.5), loc="center left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def pcv_img_histogram(img: np.ndarray):
    """
    """
    # PlantCV expects RGB for color analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask: analyze the whole image
    mask = np.ones(img_rgb.shape[:2], dtype=np.uint8) * 255

    pcv.params.debug = "print"
    pcv.params.debug_outdir = "./debug"

    # Generate color histogram plot for all color spaces
    hist_plot = pcv.analyze.color(
        rgb_img=img_rgb,
        labeled_mask=mask,
        n_labels=1,
        colorspaces="all"
    )


def pcv_pseudolandmarks(img: np.ndarray):
    """
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    mask = pcv.threshold.binary(
        gray_img=s,
        threshold=60,
        object_type='light'
    )
    mask = pcv.fill(bin_img=mask, size=300)
    mask = pcv.median_blur(mask, ksize=3)

    pcv.params.text_size = 0
    #pcv.params.debug_out = "./temp_debug"
    #pcv.params.debug = "print"
    pcv.params.debug = "print"
    pcv.params.debug_outdir = "./debug"

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)

    debug_dir = pcv.params.debug_outdir

    files = glob.glob(os.path.join(debug_dir, "*"))
    if not files:
        raise FileNotFoundError("No debug image was written by PlantCV.")

    latest_file = max(files, key=os.path.getmtime)

    debug_img = cv2.imread(latest_file)
    if debug_img is None:
        raise ValueError(f"Failed to read debug image: {latest_file}")

    #annotated = draw_pseudolandmarks(img, top, bottom, center_v, radius=4)

    return debug_img

    '''dot_img = pcv.homology.visualize_pseudolandmarks(img=img, obj=obj, mask=mask, 
                                        top=left, bottom=right, center_v=center_h)
    '''
    #plotted_img = pcv.outputs.debug_images[-1]

    #return plotted_img

    '''vis = pcv.visualize.pseudolandmarks(
        img=img,
        top=top,
        bottom=bottom,
        center_v=center_v
    )'''

    #top, bottom, center_v = pcv.analyze.y_axis_pseudolandmarks(img=img, mask=mask)
    
    '''contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obj = max(contours, key=cv2.contourArea) if contours else None

    if obj is not None:
        # This function actually DRAWS the dots onto the image
        landmark_img = pcv.homology.pseudolandmarks(
            img=img, 
            obj=obj, 
            mask=mask, 
            top=top, 
            bottom=bottom, 
            center_v=center_v
        )
        return landmark_img
    '''
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Take the largest contour (the leaf)
    ##if len(contours) > 0:
    #    obj = max(contours, key=cv2.contourArea)
    #else:
    #    return img # Return original if no object found

    # 1. Identify the object (contour) from the mask
    # This is required for homology functions
    #obj, mask = pcv.object_composition(img=img, contours=[mask])

    # 2. Generate Pseudolandmarks
    # 'y_axis_pseudolandmarks' divides the leaf into horizontal slices
    # 'sections' determines how many points are generated (e.g., 20)
    #top, bottom, center_v = pcv.homology.y_axis_pseudolandmarks(
    #    img=img, 
    #    obj=obj, 
    #    mask=mask, 
    #    label=""
    #)

    '''landmark_img = pcv.visualize.pseudolandmarks(
        img=img, 
        obj=obj, 
        mask=mask, 
        top=top, 
        bottom=bottom, 
        center_v=center_v
    )'''

    #return img


def pcv_analyze(img: np.ndarray):
    """
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    mask = pcv.threshold.binary(
        gray_img=s,
        threshold=60,
        object_type='light'
    )
    mask = pcv.fill(bin_img=mask, size=300)
    mask = pcv.median_blur(mask, ksize=3)

    pcv.params.text_size = 0

    shape_img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1, label="")
    
    return shape_img


def pcv_roi(img: np.ndarray):
    """
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    # define green range
    lower_green = (35, 50, 50)
    upper_green = (85, 255, 255)

    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)


    mask = pcv.threshold.binary(
        gray_img=s,
        threshold=60,
        object_type='light'
    )
    mask = pcv.fill(bin_img=mask, size=300)
    mask = pcv.median_blur(mask, ksize=3)

    healthy_mask = cv2.bitwise_and(healthy_mask, mask)

    # --- STEP 3: Keep only largest object (removes noise) ---
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #largest = max(contours, key=cv2.contourArea)

    #clean_mask = np.zeros_like(mask)
    #cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    # --- STEP 4: Create green overlay ---
    overlay = img.copy()
    overlay[healthy_mask > 0] = (0, 255, 0)  # green

    # --- STEP 5: Blend for visualization ---
    result = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)

    # get image size
    h, w = result.shape[:2]

    # draw rectangle (blue border)
    cv2.rectangle(
        result,
        (0, 0),        # top-left corner
        (w-1, h-1),    # bottom-right corner
        (255, 0, 0),   # blue (BGR)
        3              # thickness
    )

    return result


def pcv_mask(img: np.ndarray):
    """
    """
    # Convert RGB to HSV
    #hsv = pcv.rgb2hsv(rgb_img=img)

    # Extract saturation channel (good for separating leaf from background)
    #s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    # Convert RGB → HSV using OpenCV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract saturation channel
    s = hsv[:, :, 1]

    # Threshold saturation
    mask = pcv.threshold.binary(
        gray_img=s,
        threshold=60,
        object_type='light'
    )

    

    #mask = pcv.median_blur(img=mask, ksize=5)
    #mask = pcv.fill(bin_img=mask, size=500)
    # remove small holes
    mask = pcv.fill(bin_img=mask, size=300)
    #mask = pcv.fill_holes(mask)

    # remove salt-pepper noise
    mask = pcv.median_blur(mask, ksize=3)
    
    pcv.plot_image(mask)

    # Apply mask to image
    masked = pcv.apply_mask(
        img=img,
        mask=mask,
        mask_color='white'
    )
    return masked

def pcv_gaussian_blur(img: np.ndarray):
    """
    """
    # Extract saturation channel
    img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    img = pcv.threshold.otsu(img, object_type='light')
    #img = pcv.threshold.binary(gray_img=img, threshold=251, object_type='light')

    img = pcv.gaussian_blur(img=img, ksize=(3,3), sigma_x=0, sigma_y=None)
    #img = pcv.fill(img, size=1)

    # Threshold
    '''img = pcv.threshold.binary(
        gray_img=img,
        threshold=120,
        object_type='light'
    )'''

    '''img = pcv.rgb2gray(img)

    img = pcv.threshold.gaussian(
        gray_img=img,
        ksize=231,
        offset=25,
        object_type="dark"
    )'''

    #img, obj = pcv.fill_objects(bin_img=img, size=500)

    '''img = pcv.rgb2gray_hsv(img, channel='h')

    img = pcv.threshold.binary(
        gray_img=img,
        threshold=120,
        object_type='dark'
    )'''
    '''img = pcv.rgb2gray(rgb_img=img)

    img = pcv.threshold.otsu(
    gray_img=img,
    object_type="dark"
    )'''

    '''img = pcv.threshold.gaussian(
    gray_img=img,
    ksize=251,
    offset=35,
    object_type="dark"
    )'''

    # 1. Apply Gaussian Blur (as noted in the Figure label)
    #img = pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0, sigma_y=0)

    # 2. Extract the Saturation channel 
    # This channel usually provides the best contrast for leaf/background separation
    #img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    # 3. Apply the Binary Threshold
    # This is the function that creates Figure IV.2
    # Adjusting 'threshold' will change the density of the white areas
    #img = pcv.threshold.binary(gray_img=img, threshold=95, object_type='light')

    #if len(img.shape) == 3:
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #pcv.params.debug = "print"
    #img = pcv.threshold.gaussian(gray_img=img, ksize=50, offset=15,object_type='dark')

    #img = pcv.threshold.binary(gray_img=img, threshold=36, object_type='dark')

    #img = pcv.threshold.otsu(gray_img=img, object_type='dark')

    #img = pcv.threshold.texture(img, ksize=6, threshold=7, offset=3, texture_method='dissimilarity', borders='nearest')

    #img = pcv.gaussian_blur(img=img, ksize=(9,9), sigma_x=0, sigma_y=None)

    # laplacian edge filter
    #img = pcv.laplace_filter(gray_img=img, ksize=1, scale=1)

    # convert to binary image
    #img = pcv.threshold.binary(gray_img=img, threshold=30, max_value=255, object_type='light')
    '''img = pcv.threshold.binary(
    gray_img=img,
    threshold=30,
    object_type='light'
    )'''

    return img


def gaussian_blur(img: np.ndarray):
    """
    """
    #img = img[:,:,1]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Convert to HSV color space
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 3: Define the Green range
    # Note: In OpenCV, Hue is 0180. Green is roughly 3585.
    #lower_green = np.array([35, 40, 40])
    #upper_green = np.array([90, 255, 255])

    # Step 4: Create the Binary Mask (Your Figure IV.2)
    #img = cv2.inRange(img, lower_green, upper_green)
    #lower_bg = np.array([0, 0, 100])      # Low saturation, mid brightness
    #upper_bg = np.array([180, 50, 255])   # High brightness

    #img = cv2.inRange(img, lower_bg, upper_bg)
    # laplacian filter
    #img = cv2.Laplacian(img, cv2.CV_64F)
    #lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)

    # convert to absolute scale
    #img = cv2.convertScaleAbs(img)

    # threshold
    #_, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    #img = cv2.Canny(img, 50, 150)

    #_, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # 4. Optional: invert colors (to match white leaf on black)
    #result = cv2.bitwise_not(thresh)
    # Otsu threshold (auto threshold selection)
    '''_, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )'''

    return img



def  transformation(filepath: Path, dirpath: Path)-> None:
    """
    """
    #current_dir = dirpath.parent
    current_dir = dirpath
    print(current_dir)

    filename = filepath.stem
    print(filename)

    img = cv2.imread(str(filepath))
    
    transformations = {
        #"Flip": flip(img),
        #"Rotate": rotate(img),
        "pcv_pseudolandmarks": pcv_pseudolandmarks(img),
        "pcv_analyze": pcv_analyze(img),
        "pcv_roi": pcv_roi(img),
        "pcv_mask": pcv_mask(img),
        #"gaussian_blur": gaussian_blur(img)
        "pcv_gaussian_blur": pcv_gaussian_blur(img)
        
        
    }

    for trn_name, trn_img in transformations.items():
        save_path = current_dir / f"{filename}_{trn_name}.JPG"
        cv2.imwrite(str(save_path), trn_img)
        #pcv_img_histogram(trn_img)
        plot_leaf_color_histogram(trn_img)


def main():
    """main()"""

    try:
 
        '''
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        filepath = Path(sys.argv[1])
        print(filepath)

        is_image_file(filepath)
        transformation(filepath)'''
        parser = argparse.ArgumentParser()

        parser.add_argument("-src", required=True)
        parser.add_argument("-dst", required=True)
        #parser.add_argument("mask", action="store_true")

        args = parser.parse_args()

        print(args.src)
        print(args.dst)
        #print(args.mask)

        #if args.src and mask:
        filepath = Path(args.src)
        dirpath = Path(args.dst)

        is_path_dir(dirpath)
        is_image_file(filepath)

        transformation(filepath, dirpath)



    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
