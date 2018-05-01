# Image-Classification
Classifying the paintings based on their styles. 

Later we will move further to classify the paintings based on painters or more specific styles.

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
### Logistic classification

Use crawler to download images

Download lib files from https://www.mathworks.com/matlabcentral/fileexchange/55863-logistic-regression-for-classification

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

We use CNN model to do training on Realism and Abstract-art. The model structure is listed as below:

![example](https://oh1ulkf4j.qnssl.com/example.png)

The Training steps are:

![](https://oh1ulkf4j.qnssl.com/15251608156302.jpg)

The final result is:

| Accuracy (Total number)  | CNN  |
|---|---|
| Training accuracy (1701)  | 93.5%  | 
| Validation Accuracy (323)  | 86.0% (279)  |  
| Testing Accuracy (221)  | 86.4% (191)  | 









