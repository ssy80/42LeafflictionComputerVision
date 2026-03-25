import cv2
import sys
import os
import numpy as np
from pathlib import Path
import argparse
from utils import is_path_dir, is_image_file
from plantcv import plantcv as pcv
import glob
import matplotlib.pyplot as plt
import shutil


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
        mask = np.full(img.shape[:2], 255, dtype="uint8") # 2d mask of 255 values (white)

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


def get_max_width(center_v):
    # 1. Flatten the list of points into a simple (N, 2) array
    # This turns [(1,2), (1,2)...] into one big table of X and Y
    all_pts = np.squeeze(np.array(center_v))
    
    # 2. Group by Y-coordinate
    # We find all unique Y values (the "slices")
    unique_ys = np.unique(all_pts[:, 1])
    
    max_w = 0
    for y in unique_ys:
        # Get all X coordinates for this specific Y slice
        x_coords = all_pts[all_pts[:, 1] == y][:, 0]
        
        if len(x_coords) >= 2:
            # Width is Max X minus Min X at this height
            w = np.ptp(x_coords) # ptp = "peak to peak" (max - min)
            if w > max_w:
                max_w = w
                
    return max_w


def pseudolandmarks(img: np.ndarray, leaf_features: dict):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small holes
    Remove salt-pepper noise
    Use pseudolandmarks x-axis
    Return the image created
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5,5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')
    #mask = pcv.fill_holes(mask)            
    mask = pcv.fill(mask, size=50)
         
    mask = pcv.median_blur(mask, ksize=3)
    
    pcv.params.text_size = 0
    pcv.params.debug = "print"
    pcv.params.debug_outdir = "./debug"
    debug_dir = pcv.params.debug_outdir

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
    else:
        os.makedirs(debug_dir)

    pcv.outputs.clear()

    top, bottom, center_v = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)

    files = glob.glob(os.path.join(debug_dir, "*")) # only 1 file
    debug_img = cv2.imread(files[0])

    # Extract features
    #leaf_features["relative_width"] = relative_width
    return debug_img


def analyze(img: np.ndarray, leaf_features: dict):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small holes
    Remove salt-pepper noise
    use plantcv analyze
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5,5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')
    #mask = pcv.fill_holes(mask)            
    mask = pcv.fill(mask, size=50)         
    mask = pcv.median_blur(mask, ksize=3)
    
    pcv.params.text_size = 0
    pcv.outputs.clear()

    shape_img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1, label="leaf_data")
    
    # Extract features

    leaf_observations = pcv.outputs.observations['leaf_data_1']  # Get the dictionary for this specific label
    print(leaf_observations.keys())

    area = leaf_observations['area']['value']
    solidity = leaf_observations['solidity']['value']
    perimeter = leaf_observations['perimeter']['value']
    ellipse_eccentricity = leaf_observations['ellipse_eccentricity']['value']
    longest_path = leaf_observations['longest_path']['value']
    width = leaf_observations['width']['value']
    height = leaf_observations['height']['value']
    #convex_hull_perimeter = leaf_observations['convex_hull_perimeter']['value']
    convex_hull_area = leaf_observations['convex_hull_area']['value']
    relative_width = width / height
    circularity = (4 * 3.14159 * area) / (perimeter**2)
    relative_perimeter = perimeter / area

    #convexity = convex_hull_perimeter / perimeter
    extent = area / (width * height)
    
    pcv.outputs.clear()
    
    # Convert image to LAB color space for scientific greenness
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Use the mask to "select" only leaf pixels
    # mask > 0 creates a list of True/False coordinates
    leaf_pixels_a = lab_img[:, :, 1][mask > 0]
    
    # Calculate the Mean Greenness (The 'a' channel)
    # A very low negative number (e.g., -20) means very green. A number near 0 or positive means brown/gray.
    mean_greenness_a = np.mean(leaf_pixels_a) if leaf_pixels_a.size > 0 else 0
    std_greenness_a = np.std(leaf_pixels_a) if leaf_pixels_a.size > 0 else 0


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rugosity = 0
    hull_ratio = 0

    if contours:
        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)
        
        # 2. Calculate the Convex Hull
        hull = cv2.convexHull(cnt)
        
        # 3. Calculate the Perimeter of that Hull
        # True means the curve is closed
        convex_hull_perimeter = cv2.arcLength(hull, True)

        # 4. Create the "Rugosity" feature
        # A value of 1.0 means a perfectly smooth oval. 
        # A value of 0.7 means a jagged/decaying leaf edge.
        rugosity = convex_hull_perimeter / perimeter
        hull_ratio = perimeter / convex_hull_area
    
    leaf_features["relative_perimeter"] = relative_perimeter
    leaf_features["solidity"] = solidity
    #leaf_features["perimeter"] = perimeter
    leaf_features["ellipse_eccentricity"] = ellipse_eccentricity
    leaf_features["mean_greenness_a"] = mean_greenness_a
    leaf_features["std_greenness_a"] = std_greenness_a
    leaf_features["relative_width"] = relative_width
    leaf_features["longest_path"] = longest_path
    leaf_features["circularity"] = circularity
    #leaf_features["convexity"] = convexity
    leaf_features["extent"] = extent
    leaf_features["rugosity"] = rugosity
    leaf_features["hull_ratio"] = hull_ratio
    return shape_img


def roi(img: np.ndarray, leaf_features: dict):
    """
    Get HSV from BGR
    Extract saturation (s) channel
    Define green range
    Get healthy mask from green range
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small holes
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

    blur = pcv.gaussian_blur(img=s, ksize=(5,5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')
    #mask = pcv.fill_holes(mask)            
    mask = pcv.fill(mask, size=50)         
    mask = pcv.median_blur(mask, ksize=3)
    
    # Only keep Green pixels that are actually INSIDE the leaf
    final_healthy_mask = cv2.bitwise_and(healthy_mask, mask)

    overlay = img.copy()
    overlay[final_healthy_mask > 0] = (0, 255, 0)  # green BGR

    h, w = overlay.shape[:2]

    cv2.rectangle(
        overlay,
        (0, 0),        # top-left corner
        (w-1, h-1),    # bottom-right corner
        (255, 0, 0),   # blue (BGR)
        3              # thickness
    )

    # Extract features

    total_area = np.count_nonzero(mask)
    healthy_area = np.count_nonzero(final_healthy_mask)

    # Calculate Features
    healthy_ratio = healthy_area / total_area if total_area > 0 else 0
    disease_area = total_area - healthy_area
    disease_ratio = disease_area / total_area if total_area > 0 else 0
    
    # Count individual spots
    num_spots, _ = cv2.connectedComponents(cv2.subtract(mask, final_healthy_mask))
    spot_density = num_spots / total_area

    lesion_size = disease_area / num_spots
    #infection_pressure: num_spots / perimeter

    leaf_features["healthy_ratio"] = healthy_ratio
    #leaf_features["disease_area"] = disease_area
    #leaf_features["disease_ratio"] = disease_ratio
    leaf_features["spot_density"] = spot_density
    leaf_features["lesion_size"] = lesion_size

    return overlay


def mask(img: np.ndarray, leaf_features: dict):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small holes
    Remove salt-pepper noise
    Apply mask to image
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5,5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')

    #mask = pcv.fill_holes(mask)            # removes "holes" from inside the leaf (black spots on white).
    mask = pcv.fill(mask, size=50)         # removes "trash" from the background (white specks on black).
    mask = pcv.median_blur(mask, ksize=3)
    
    masked = pcv.apply_mask(
        img=img,
        mask=mask,
        mask_color='white'
    )

    # Extract features
    #leaf_features["mask"] = "mask"
    return masked


def gaussian_blur(img: np.ndarray, leaf_features: dict):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5,5), sigma_x=0, sigma_y=None)

    threshold_img = pcv.threshold.otsu(blur, object_type='light')

    # Extract features
    #leaf_features["gaussian_blur"] = "gaussian_blur"
    return threshold_img


def original(img: np.ndarray, leaf_features: dict):
    """
    return original image
    """
    # Extract features
    #leaf_features["original"] = "original"
    return img


def transformation(filepath: Path)-> None:
    """
    Various transformation to a single image
    Extract features from various transformations
    """
    img = cv2.imread(str(filepath))
    leaf_features = {}

    transformations = {
        "original": original(img.copy(), leaf_features),
        "gaussian_blur": gaussian_blur(img.copy(), leaf_features),
        "mask": mask(img.copy(), leaf_features),
        "roi": roi(img.copy(), leaf_features),
        "analyze": analyze(img.copy(), leaf_features),
        "pseudolandmarks": pseudolandmarks(img.copy(), leaf_features)
    }

    print(leaf_features)

    return transformations, leaf_features


def  transform_dir(src_path: Path, dest_path: Path)-> None:
    """
    Loop the src directory
    Transform every image in the src directory
    Save to dst directory
    """
    for file_path in src_path.glob("*.JPG"):
        filename = file_path.stem
        transformed, _ = transformation(file_path)
        for trn_name, trn_img in transformed.items():
            save_path = dest_path / f"{filename}_{trn_name}.JPG"
            cv2.imwrite(str(save_path), trn_img)
            

def main():
    """main()"""

    try:
 
        parser = argparse.ArgumentParser()

        parser.add_argument("-src", required=True)
        parser.add_argument("-dst", required=False)
        
        args = parser.parse_args()
        filepath = Path(args.src)
        
        dirpath = None
        if args.dst:
            dirpath = Path(args.dst)

        if dirpath:                                 # dir src, dir dst
            is_path_dir(dirpath)
            is_path_dir(filepath)
            transform_dir(filepath, dirpath)
        else:                                       # single image file src
            is_image_file(filepath)
            transformed, _ = transformation(filepath)
            for _, trn_img in transformed.items():
                pcv.plot_image(trn_img)
                #plot_leaf_color_histogram(trn_img)
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
