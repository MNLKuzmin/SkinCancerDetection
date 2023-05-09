# Reproducibility

In this file I am going to show you how to reproduce my project,
where to get the data, what packages to install, where to find the project on Colab and more.
### Projects: 

My project is both in a Jupyter Notebook that you can find on [my Github](https://github.com/MNLKuzmin/SkinCancerDetection) and on Google Colab at the following [link](https://drive.google.com/drive/folders/1-yAUnecUr5Pwet6SfvPAliKE9X9sFVcj) --- RIGHT NOW NOT PUBLIC FIX IT
### Data:
The dataset was taken from Kaggle at the following [link](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).
If you dont have an account you will need to create it create a sign in - it's easy and free and Kaggle has a lot of very useful datasets to work on.
The data can simply be downloaded using the top right black button "Download". The file is 2GB in size.
One the file is downloaded as a zip file it will need to be unzipped.

Once the dataset is unzipped, it is divided already into 2 folders, 'Train' and 'Test' which have each 9 subfolders with the 9 different types of skin anomalies.
Both the Jupyter notebook and Google Colab project have instructions to load the images from the folders as they are.
What will be required is only to download the folder and have them in the same folder as the project, both in Jupyter and in Colab.

In the second part of the project the images are separated into 2 classes instead of 9, and this is done by creating a separate folder, 'binary', with subfolders 'cancerous' and 'benign' each divided also into 'train' and 'test'.
The code to create such directories and copy the images in the folders is included both in Jupyter notebook and in Google Colab, it just needs to be uncommented and ran only once.

### Environment:

The requirements for the environment are listed in a separate document, available at the following [link]

For the project in Google Colab to run it was necessary to install a few packages, those are included in the very first cell of code in the Colab Project.

### For More Information

Please review my full analysis in [my Jupyter Notebook](./SkinCancerDetection.ipynb) or my [presentation](./Presentation.pdf).

For any additional questions, please contact **Maria Kuzmin, marianlkuzmin@gmail.com**
