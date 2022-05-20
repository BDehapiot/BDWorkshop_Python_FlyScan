from pathlib import Path
print(Path.cwd())

im_path = Path('fly_images')

print(im_path) # relative path
print(im_path.resolve()) # full path
print(Path('fly_images').resolve()) # full path

im_paths = [] # create an empty list
for path in Path('fly_images').iterdir():
    im_paths.append(path)
    
print(im_paths[0]) # relative path 
print(im_paths[0].resolve()) # full path

print(im_paths[0].name) # name
print(im_paths[0].stem) # name without extension
print(im_paths[0].suffix) # extension only

#%%

from skimage import io
im_first = io.imread(im_paths[0])

print(type(im_first))

print(im_first.shape) # 2 elements tuple
im_height = im_first.shape[0]
im_width = im_first.shape[1]

im_count = len(im_paths)
print(im_count)

import numpy as np

# 1) create a zero array with shape (im_count, im_height, im_width)
# 2) fill this array with imported images

# correction
im_all = np.zeros([im_count, im_height, im_width], dtype='uint8')
for i, path in enumerate(im_paths):   
    im_all[i,...] = io.imread(path)  
    
import napari
viewer = napari.view_image(im_all)

import matplotlib.pyplot as plt

# compute min and max projections
im_all_min = np.min(im_all, axis=0)
im_all_max = np.max(im_all, axis=0)

#%%

# display results with matplotlib
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
ax[0].imshow(im_all_min, cmap='gray')
ax[0].set_title('min projection')
ax[1].imshow(im_all_max, cmap='gray')
ax[1].set_title('max projection')
plt.show()

# 1) compute median projection (im_all_median)
# 2) subtract median projection from raw images (im_all_sub)
# 3) display result for the first image in matplotlib

# Correction

# compute median projection
im_all_median = np.median(im_all, axis=0)

# subtract median projection from raw images
im_all_sub = np.subtract(im_all, im_all_median)

# display result for the first image in matplotlib
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))
ax[0].imshow(im_all[0], cmap='gray')
ax[0].set_title('raw')
ax[1].imshow(im_all_median, cmap='gray')
ax[1].set_title('median projection')
ax[2].imshow(im_all_sub[0], cmap='gray')
ax[2].set_title('raw - median')
plt.show()

from skimage.util import invert

im_all_sub = invert(im_all_sub)
plt.imshow(im_all_sub[0], cmap='gray')
plt.show()

from skimage.filters import threshold_yen

thresh = threshold_yen(im_all_sub[0])
im_all_mask = im_all_sub > thresh
print(thresh)

plt.imshow(im_all_mask[0], cmap='gray')
plt.show()

plt.imshow(im_all_mask[0,275:375,40:140], cmap='gray')
plt.show()

from skimage.filters import gaussian
from skimage.morphology import remove_small_objects

im_all_mask = gaussian(im_all_sub, 2, channel_axis=0) > thresh
im_all_mask = remove_small_objects(im_all_mask, min_size=50)

plt.imshow(im_all_mask[0,275:375,40:140], cmap='gray')
plt.show()

from skimage.morphology import dilation

im_all_display = np.zeros_like(im_all)

for i, mask in enumerate(im_all_mask):    
    outlines = dilation(mask) ^ mask
    outlines = outlines.astype('uint8')*255 
    im_all_display[i,...] = np.maximum(im_all[i,...], outlines)
    
viewer = napari.view_image(im_all_display)

def im_segment(im_all, thresh_coeff=1.0, gaussian_sigma=2.0, min_size=50):
        
    # 1) subtract static background
    # 2) get binary mask
    # 3) make a display 
    
    ...
    
    return im_all_mask, im_all_display 

# Correction

def im_segment(im_all, thresh_coeff=1.0, gaussian_sigma=2.0, min_size=50):
    
    # subtract static background
    im_all_median = np.median(im_all, axis=0)
    im_all_sub = np.subtract(im_all, im_all_median)
    im_all_sub = invert(im_all_sub)

    # get binary mask
    im_all_mask = gaussian(im_all_sub, 2, channel_axis=0) > thresh*thresh_coeff
    im_all_mask = remove_small_objects(im_all_mask, min_size=50)
    
    # make a display
    for i, mask in enumerate(im_all_mask): 
        outlines = dilation(mask) ^ mask
        outlines = outlines.astype('uint8')*255 
        im_all_display[i,...] = np.maximum(im_all[i,...], outlines)
        
    return im_all_mask, im_all_display  

im_all_mask, im_all_display = im_segment(im_all, thresh_coeff=1.0, gaussian_sigma=2.0, min_size=50)
viewer = napari.view_image(im_all_display)

#%%

from skimage.morphology import label

im_all_labels = np.zeros_like(im_all)
for i, mask in enumerate(im_all_mask):       
    im_all_labels[i,...] = label(mask)
    
max_id = 0
for i, labels in enumerate(im_all_labels):
    labels[labels != 0] += max_id
    
    if np.max(labels) != 0:
        max_id = np.max(labels)
    
    im_all_labels[i,...] = labels

viewer = napari.view_labels(im_all_labels) # view_labels instead of view_image

from skimage.measure import regionprops

im_all_props = []

for props in regionprops(im_all_labels):   
    
    temp_props = {       
        'label': props['label'],
        'centroid': props['centroid'],
        'area': props['area'],       
        }
    
    im_all_props.append(temp_props)

for props in im_all_props:        
    print(props['centroid'])    
