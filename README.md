This project was created as part of my application at sipgate.
It uses the deep learning architecture 'U-Net' (see image file in this repository) to transform segmented pictures of building facades into 'photos' of facades.
The dataset is a preprocessed version of the pix2pix set: https://www.tensorflow.org/tutorials/generative/pix2pix?hl=en

How to use:
- Run Main.py
- To re-train the model, delete the 'trainedUnet.pth' before running the project
    - The names of the dataset folders and the training parameters can be configures in Settings.py
    - The architecture and dataset are optimized for training on a NVIDIA GPU with 8 GB VRAM. If you encounter problems, enforce Training on the CPU in Settings.py and reduce the epochs, as training without CUDA takes a long time
	- After training, a logfile and training diagram will be created to optimize training behavior
- This program will train on the images in the Dataset/Input and Dataset/Target folders. After training, it will generate predictions for the whole dataset and save them in Dataset/Generated by default. Existing predictions will be overwritten.