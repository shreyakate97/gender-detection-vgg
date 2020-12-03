# gender-detection-vgg
Gender detection using transfer learning

A. Model conversion from Caffe to Tensorflow using tf.keras

1. Install Caffe using the following command
!apt install caffe-cpu

2. Run the follwing command to get the weight converter (caffe-keras) file. 
!wget  https://raw.githubusercontent.com/pierluigiferrari/caffe_weight_converter/master/caffe_weight_converter.py

I have included the caffe_weight_converter.py file as well in the folder. I got this code from https://github.com/pierluigiferrari/caffe_weight_converter.

3. Run model_conversion.py to get the contents of VGG_FACE_deploy.prototxt file and view the neural network architecture information in it.

4. Run the following command to store the weights from VGG_FACE.caffemodel in keras_weights.h5 file 

!python caffe_weight_converter.py 'keras_weights' \
                                 'VGG_FACE_deploy.prototxt' \
                                 'VGG_FACE.caffemodel' \
                                 --verbose

6. The weights have been saved in keras_weights.h5. I have uploaded this file in the folder. 
##################################################################################################################
B. Transfer Learning- Create, Train and Evaluate model : model_trainandeval.py
I have used the above CNN as a feature descriptor to build a classifier for the gender dataset.

1. I wrote the code for model architecture in tf.keras and loaded the pretrained model weights from keras_weights.h5 to it. (I omitted the fc8 layer, which was the classifier layer in the original model)

2. Next, I made all the layers non-trainable and added a binary classifier layer to the model. I printed out the model summary.

3. Dataset rearranging and loading to keras
To arrange the dataset into 2 classes (female and male), I wrote the code which is in data_parsing.py. I ran this code inside the combined folder of the dataset.
I have included the final data I used in the folder. I loaded this data to keras with a batch_size of 128.

4. I divided the dataset into training and validation with an 80/20 splitsince this gave me enough data to both train and validate/test the model. (Total files: 33118, Training: 26495, Validation: 6623)

5. I compiled and fit the model with SparseCategoricalLoss(from_logits=True), adam optimizer and over 15 epochs.

6. I created a final_tensorflow_model folder and saved my model in it. I have included this in the folder.

7. I evaluated the model on the validation data and got an accuracy of 86.2%

8. The code for steps in B. is in model_trainandeval.py

C. Metrics

1. I plotted the accuracy and loss vs no. of epochs curves giving the performance over both classes.
2. I have printed the classification report and confusion matrix which give information of each class separately. 
class '0' : female, class '1': male
Confusion matrix: total validation data = 6623
Number of female images = (2670+486) = 3156 Number of male images = (428+3039) = 3467
Classification accuracy for females = 100(2670/3156) = 84.6% 
Classification accuracy for males = 100(3039/3467) = 87.65%
Therefore, we get the total accuracy as 86.2% (We got this value for model.evaluate())

3. I have printed out some images that are misclassified. I noticed that the misclassification was in images of children and where the images were unclear.

4. The code for this is in model_trainandeval.py as well.


I have included the Google Colab notebook where I have run my code and shown results. The link is:
https://colab.research.google.com/drive/1O-fQMvPVPyNT9tuuq4F6vuDA13Qp-6Cv?usp=sharing
