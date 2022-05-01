#%%
from pathlib import Path

# Show path to current directory 
print(Path.cwd())

# Check if fly_chamber directory exists
if Path('fly_chamber').exists():
    print('You are in the good directory')
else:
    print('You are NOT in the good directory')

# List image paths in fly_chamber directory
image_paths = []
for path in Path('fly_chamber').iterdir():
    image_paths.append(path)
    print(path.name) # print image names
    
#%%
import napari
import numpy as np  
from skimage import io

# Import first image
im_first = io.imread(image_paths[0])

# Check im_first shape
print(im_first.shape)
im_height = im_first.shape[0]
im_width = im_first.shape[1]

# Count images in fly_chamber directory
im_count = len(image_paths)

# Import all images as a 3D Numpy array
im_all = np.zeros([im_count, im_height, im_width], dtype='uint8')
for i, path in enumerate(image_paths):   
    im_all[i,...] = io.imread(path)  

# Display im_all in Napari
# viewer = napari.view_image(im_all)
     
#%%
import matplotlib.pyplot as plt
from skimage.util import invert

# Substract static background
im_all_median = np.median(im_all, axis=0)
im_all_sub = im_all - im_all_median
im_all_sub = invert(im_all_sub) # invert images

# Display results in matplotlib
fig, ax = plt.subplots(nrows=3, ncols=1, dpi=300, constrained_layout=True)
ax[0].imshow(im_all[0,...], cmap='gray')
ax[0].set_title('raw')
ax[0].axis('off')
ax[1].imshow(im_all_median, cmap='gray')
ax[1].set_title('median')
ax[1].axis('off')
ax[2].imshow(im_all_sub[0,...], cmap='gray')
ax[2].set_title('raw - median')
ax[2].axis('off')

# Display im_all_sub in Napari
# viewer = napari.view_image(im_all_sub)

#%%
from skimage.filters import try_all_threshold, threshold_yen

# Try different thresholding methods
fig, ax = try_all_threshold(im_all_sub[80,...])
plt.show()

# Binarize im_all_sub using yen thresholding method
thresh = threshold_yen(im_all_sub[0,...])
im_all_mask = im_all_sub > thresh

# Display im_all_mask in Napari
# viewer = napari.view_image(im_all_mask)

#%%
from skimage.segmentation import clear_border
from skimage.morphology import dilation, remove_small_objects

# Clean the mask
tmp = im_all_mask[80]
im_all_mask = remove_small_objects(im_all_mask, min_size=150)
# im_all_mask = dilation(im_all_mask)
# im_all_mask = clear_border(im_all_mask)

# Display results in matplotlib
fig, ax = plt.subplots(nrows=2, ncols=1, dpi=300, constrained_layout=True)
ax[0].imshow(tmp[110:260,80:230], cmap='gray')
ax[0].axis('off')
ax[1].imshow(im_all_mask[80,110:260,80:230], cmap='gray')
ax[1].axis('off')

#%%
 
# io.imsave(
#     Path.cwd()/'im_all_sub.tif', 
#     im_all_sub.astype('float32'), 
#     check_contrast=False
#     ) 
