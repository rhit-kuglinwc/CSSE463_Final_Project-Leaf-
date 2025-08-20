# CSSE463_Final_Project-Leaf-
The repository for the final project of CSSE 463 that derives tree species from leaves.

## There are two `.mat` files on the gebru server
- `alexnet.mat`: contains the alexnet cnn that was made with transfer learning
- `xception.mat`: contains the xception cnn that was made from transfer learning


## DEMO.m
This file contains the code that was used for the demo during our presentation. To run this file you will need to have the demo dataset root directory which was made from taking six images from the validate and test sets and putting them into a sub directory of plants (2). Once this has been done you can run the sections in sequential order and at the end you will get a figure of the images that you have selected the titles are not hard coded but the image reads are.

## get_image_data.m
This file is where we augment the images that we are given. It follows the subdirectory structure of plants (2). It also uses code that was made with the help of chatGPT to take only the tag of each of the images. If using a different pretrained network change the net on line 23.

## xception_cnn_transfer_learning.m
This file is where the xception cnn transfer learning happened. You need to run `get_image_data.m` first to run this file. This file was where the original alexnet model was run but because this model was so much better it replaced the alexnet code. For this model the test accuracy was done on the command line.

## all_leaf_fitcecoc.m
This file is where the grid search for the baseline fitceoc model is done. You need to run `get_image_data.m` first to run this file. For this model the test accuracy was done on the command line.
