# Painting-Classification
This project classifies paintings based on their styles. 

We use deep learning (cnn), transfer learning (vgg-16) and Logistic regression to do the style classification. 

If you want to start from beginning, please read the following steps, which will help!

Feel free to reach out us if you encounter any problem:
+ Tianyu Hong (hongty106@gmail.com): Data collection, deep learning, transfer learning
+ Xiuyang Duan (xiuyangd@andrew.cmu.edu): logistic regression, image feature extracting


## Data 
### Collecting
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

### Preprocessing
Use the script `deep_learning/image_preprocessing` to do image preprocessing is you want to use CNN model or transfer learning. </br> 
It will crop images to square and resize them into 100*100 pixels. 

## Classification methods
### Logistic regression

Download lib file from https://www.mathworks.com/matlabcentral/fileexchange/55863-logistic-regression-for-classification

Create a new folder 'lib' in Painting-Classification/HOG/, then put file downloaded in this folder and unzip it.

Add the lib folder/files to path for Matlab.

Run main.m

![](https://oh1ulkf4j.qnssl.com/15251606251272.jpg)


### Transfer Learning
Here we use VGG_16 to generate image features. The generating as well as processing methods can be seen in `transfer_learning/features_generator.py`. </br>
And the VGG_16 features as well as the labels will be generated into `data/features/` directory.

After generating the VGG_16 features, we use validation data to choose the best kernel for SVM model and also apply Logistic regression model. The result is shown below: 

![](https://oh1ulkf4j.qnssl.com/15251604946976.jpg)


We find out that Logistic regression model is the best, so we use Logistic regression model do style classification. 

![](https://oh1ulkf4j.qnssl.com/15251605827916.jpg)
 
### Deep learning (CNN)

A tutorial of how to use our CNN model is in `deep_learning/cnn_style_classification.ipynb` <br>
You can also use `deep_learning/main.py` to do training and testing via command line. (Use `python3 main.py -h` to get instruction)<br>

We use CNN model to do training on Realism and Abstract-art. The model structure is listed as below: (*Attention: there is a typo when drawing: after the last polling, the size should be 13 x 13 x 64 instead of 12 x 12 x 64, the flatten fully connected layer should change to 10816 as well*)

![example-1](https://oh1ulkf4j.qnssl.com/example-1.png)



The Training steps are:

![](https://oh1ulkf4j.qnssl.com/15251608156302.jpg)

The final result is:

| Accuracy (Total number)  | CNN  |
|---|---|
| Training accuracy (1701)  | 93.5%  | 
| Validation Accuracy (323)  | 86.0% (279)  |  
| Testing Accuracy (221)  | 86.4% (191)  | 









