# Reproducibility

In this file I am going to show you how to reproduce my project,
where to get the data, what packages to install, where to find the project on Colab and more.
### Projects: 

My project is both in a Jupyter Notebook that you can find on [my Github](https://github.com/MNLKuzmin/SkinCancerDetection) and on Google Colab at the following [link](https://drive.google.com/drive/folders/1-yAUnecUr5Pwet6SfvPAliKE9X9sFVcj) --- RIGHT NOW NOT PUBLIC FIX IT
### Data:
The dataset was taken from Kaggle at the following [link](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).
<br>If you dont have an account you will need to create it create a sign in - it's easy and free and Kaggle has a lot of very useful datasets to work on.
<br>The data can simply be downloaded using the top right black button "Download". The file is 2GB in size, the download might take a couple of minutes depending on your download speed. Unzip both using the relevant utility for your computer (for Mac you can use [thins link](https://support.apple.com/guide/terminal/compress-and-uncompress-file-archives-apdc52250ee-4659-4751-9a3a-8b7988150530/mac),for Windows click [here](https://support.microsoft.com/en-us/windows/zip-and-unzip-files-f6dde0a7-0fec-8294-e1d3-703ed85e7ebc), and this is the link for [Linux](https://www.howtogeek.com/414082/how-to-zip-or-unzip-files-from-the-linux-terminal/)). Then after unzipping you should see a folder name Skin cancer ISIC The International Skin Imaging Collaboration. This folder has two subfolders, 'Train' and 'Test' which have each 9 subfolders with the 9 different types of skin anomalies. We recommend renaming the main folder 'Data'. Next move this folder into the data/ directory of this repository.

Both the Jupyter notebook and Google Colab project have instructions to load the images from the folders as they are.
<br>What will be required is only to download the data folder, as described above, and have them in the same directory as the project, both in Jupyter and in Colab.

In the second part of the project the images are separated into 2 classes instead of 9, and this is done by creating a separate folder, 'binary', with subfolders 'cancerous' and 'benign' each divided also into 'train' and 'test'.
<br>The code to create such directories and copy the images in the folders is included both in Jupyter notebook and in Google Colab, it just needs to be uncommented and ran only once.

### Environment:

The requirements for the environment are listed in a separate document, available at the following [link](./environment.yml) -- run conda env export environment.yml in the terminal and that creates the file

For the project in Google Colab to run it was necessary to install a few packages, those are included in the very first cell of code in the Colab Project.

This project was created on a macOS Big Sur, with CPU 2.6 GHz Dual-Core Intel Core i5 and GPU Intel Iris 1536 MB.

### For More Information

Please review my full analysis in [my Jupyter Notebook](./SkinCancerDetection.ipynb) or my [presentation](./Presentation.pdf).

For any additional questions, please contact **Maria Kuzmin, marianlkuzmin@gmail.com**
