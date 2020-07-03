import glob
import random

import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


images_path = glob.glob('Datasets/real_Image_dataset_Detection/Image/*/*')


def _make_grid(image_path_list,n_row=2, n_col=2,fig_size=(10,10)):
    assert len(image_path_list) == n_row*n_col
    fig = plt.figure(figsize=fig_size)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_row,n_col),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for ax,fname in zip(grid, image_path_list):
        # Iterating over the grid returns the Axes.
        im= plt.imread(fname)
        file_name = fname.split('/')[-1]
        ax.imshow(im)
        ax.set_title(file_name,color='blue')

    plt.show()


def object_detection_api(img_path, threshold=0.5, rect_th=2, text_size=3, text_th=3):

  boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
  img = cv2.imread(img_path) # Read image with cv2
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    # cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()
# sampling = random.choices(images_path, k=8)
# _make_grid(sampling,n_row=4, n_col=2,fig_size=(15,20))