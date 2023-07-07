# [Dependable AI] Course Project
This readme file contains a detailed description on how to run the given code and obtain the respective results.

## Instructions
To successfully run and visualize the code follow the below mentioned instructions.

### Installation
The python code requires the following packages. It is recommended to set up a conda environment first and load the packages to avoid conflicts.

* PyTorch
* NumPy
* PIL
* Torchvision
* Pandas
* Pickle
* Scipy
* tensorboardX
* h5py
* cv2
* skimage

After successfully downloading the above mentioned packages please download the submitted code file in zip format and extact the files inside.

### Dataset
Please note that the dataset is not submitted as part of the code, however database link is provided in the report, because of size limitations. 
If you are trying to reproduce the results for the research paper the CIFAR-10 dataset should be downloaded by itself using the Torchvision API. However, note that though Corrupted-CIFAR10 is uploaded in the drive link you have to manually download the Corrupted-CIFAR10 dataset and provide the necessary path to the respective code file.
Also, note that the fractals_and_fvis data is uploaded on the database link, however providing proper path to the code file to reproduce the results is in the hands of the reader.
However, the same is not the case for UCF101 dataset because of it's humongous size. You have to manually download and provide the path of the UCF101 dataset (video files) and related files to the 'train-test.py' python file.

### Run
After all the above mentioned packages, dataset and code files are successfully loaded run the 'reproduce.ipynb' file to reproduce the results mentioned in the research paper for CIFAR-10 dataset.

In order to reproduce the results for the proposed approach please run 'train-test.py' file mentioned inside the ucf101-supervised-main folder. 

Remember to change the dataset path accordingly.

Please contact the author in case of any doubts.