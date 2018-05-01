# Image-Classification
Classifying the paintings into two categories: Eastern and Western. 

Later we will move further to classify the paintings based on painters or more specific styles.

## Data collecting
The data we use in this task is from *WikiArt*. The collecting method is listed in `crawler/`. 

The collected data information is listed below: </br>

| styles  | number  | Note |
|---|---|---|
| impressionism  | 1135  | Western |
| international-gothic  | 236  | Western | 
| shin-hanga  | 420  | Eastern | 
| abstract-art  | 1109  | Western | 
| ink-and-wash-painting  | 678  | Eastern | 
| realism  | 1136  | Western | 
| northern-renaissance  | 1098  | Western | 
| gothic  |  33 | Western | 
| gongbi  | 39  | Eastern | 
| | 5884 | |

## Classification methods
### Logistic classification
Use crawler to download images
Download lib files from https://www.mathworks.com/matlabcentral/fileexchange/55863-logistic-regression-for-classification
Add the lib folder/files to path for matlab
Run main.m


### SVM classification

Here we use VGG_16 to generate image features. The generating as well as processing methods can be seen in `deep_learning/features_generator.py`. </br>
And the VGG_16 features as well as the labels will be generated into `data/features/` directory.

After generating the VGG_16 features, we use validation data to choose the best kernel for SVM model. The result is shown below:

![Screen Shot 2018-04-24 at 9.21.39 PM](https://oh1ulkf4j.qnssl.com/Screen%20Shot%202018-04-24%20at%209.21.39%20PM.png)

We achieve a 88.3% accuracy on the testing dataset.


