from matplotlib import pyplot as plt
import cv2
import numpy as np

def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : dict
        A dictionary where keys are titles and values are images.
    ncols : int
        Number of columns of subplots wanted in the display.
    nrows : int
        Number of rows of subplots wanted in the figure.
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional

def display_recommendations(reference_image, recommended_images):
    """Display the reference image and its recommended images.

    Parameters
    ----------
    reference_image : str
        Path to the reference image.
    recommended_images : list
        List of paths to recommended images.
    """
    figures = {'Reference Image': load_image(reference_image)}
    for i, img in enumerate(recommended_images):
        figures[f'Recommended Image {i+1}'] = load_image(img)
    
    plot_figures(figures, nrows=2, ncols=3)

def load_image(img_path, resized_fac=0.1):
    """Load and resize an image.

    Parameters
    ----------
    img_path : str
        Path to the image file.
    resized_fac : float
        Factor by which to resize the image.
    
    Returns
    -------
    resized : numpy.ndarray
        Resized image array.
    """
    img = cv2.imread(img_path)
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
    return resized