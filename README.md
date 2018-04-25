# Image-Classification
Classifying the paintings into two categories: Eastern and Western. 

Later we will move further to classify the paintings based on painters or more specific styles.

## Data collecting
The data we use in this task is from *WikiArt*. The collecting method is listed in `crawler/`. 

## Classification methods
### SVM classification

Here we use VGG_16 to generate image features. The generating as well as processing methods can be seen in `deep_learning/features_generator.py`. </br>
And the VGG_16 features as well as the labels will be generated into `data/features/` directory.

After generating the VGG_16 features, we use validation data to choose the best kernel for SVM model. The result is shown below:

![Screen Shot 2018-04-24 at 9.21.39 PM](https://oh1ulkf4j.qnssl.com/Screen%20Shot%202018-04-24%20at%209.21.39%20PM.png)

We achieve a 88.3% accuracy on the testing dataset.


