# EIP - 2 | Assignment - 1

#### Prajwal Prashanth | prajwal.prashanth22@gmail.com |  Batch 4 
___
## Convolution

​	A convolution is a mathematical operation on two signals/functions ( **_a_** and **_b_** ) to produce a third signal/function (**_c_**) which is determined by the magnitude of overlap of **_b_** over **_a_**, this is a general mathematical definition of convolution lets get down to specifics and talk over its application in Neural Networks which is now the state of the art model for images known as Convolution Neural Networks(CNN) .

​	In CNN we use a filters/kernels (consider **_b_**) in between two layers to convolve over previous layer (consider **_a_**) to form the next layer (**_c_**) in the network, this operation is convolution here. We operate on images (represented as matrices) instead of functions/signals, consider the example below we are convolving or moving through **_a_** : 6x6 matrix using **_b_** : 3x3 matrix to form a **_c_** : 4x4 matrix.

![alt text](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/28011851/conv.gif)

​					**Fig 1.1 :** Kernel/Filter convolving over image

_(image source: Blog by Dishashree Gupta from analyticsvidhya.com)_

___

## Filters/Kernels

​	A Filters/Kernels are the essential part of convolution layers which decides what type of filter to apply for the previous image layer these kernel check for a specific pattern or lines depending, each kernel checks for unique lines/patterns and complex combination of these will be able to classify different type of similar images from other images.

​	Kernels/Filters => **_(h , w , d) * n_**.

* **_h x w (height, width)_** : these parameters is choice made eg: 3x3 , 5x5 , 7x7 but 3x3 is used 												as it'll take less parameters to get to the equivalent of 5x5 or any other higher spatial size.

* **_d (depth)_** : this parameter is decided by how many channels are required for the next  layer

* **_n_** : this parameter is again a choice which is the no. of filters you wish to use also this decides the depth of next layer

![](https://blog.xrds.acm.org/wp-content/uploads/2016/06/Figure_2.png)

​	 	  						**Fig:1.2**:  kernel/filter 

_(image source: Abhineet Saxena from blog.xrds.acm.org)_

___

## Epochs

​	One full forward pass ( like convolutions,pooling,activations of all the layers) and one full backward pass (updating weights via backpropagation) through the neural network of the all the training set data once is known as one epoch. We don't have any certain rule to decide number of epochs we'll hope do these epochs until and stop before we start to overfit the model.

​	Usually for large data set epoch will require huge amount of ram and computing power. So, we divide the training data to numerous batches to train the network, each time we train the network with a batch we call it a iteration over the network.

​	Thus, 1 epoch = batch_size * iterations (where batch_size * iterations is complete train data)

![alt text](https://thumbs.gfycat.com/UnsteadyUnkemptKangaroo-size_restricted.gif)

​				**Fig 1.3** : Visual representation of epochs of small data set

_(image source: RimstarOrg, Youtube Channel from gfycat.com)_

___

## 1x1 Convolution

​	We use max pooling to reduce the spatial dimensions(**_w ,h_**) but for dimensionality reduction in terms of depth/features(**_d_**) we use 1x1 convolution to pool the depth and change dimensions in terms of depth without affecting change in spatial dimensions. More non-linearity by coupling these 1x1 conv with a activation function.

​	If we have a image of 100 x 100 with 20 features and want to reduce it to 5 features for next layer can be obtained by convolution of 1x1, 5 filters. Result is 100 x 100 , 5 features without any changes to the spatial dimension.

![](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

​								**Fig1.4 :** 1x1 Convolution

_(image source: Adithya Prakash from iamaaditya.github.io)_

___

## 3X3 Convolution

​	3x3 is the spatial dimension most popular and widely used as it big enough size to represent edges/lines and can represent the same receptive field of higher dimensions convolutions (5x5, 7x7, ........) with less number of weights. 3x3 conv filters reduces the spatial dimension by 2 i.e (n x n) convolved with (3x3) gives (n-2,n-2) with affecting any changes to the depth which depends on number of filters.

​	5x5 and 7x7 can replaced by 2x(3x3) , 3x(3x3) filters respectively as they have same receptive field. But there is difference in number of weights mentioned below

​			

| Dimension | Number of weights |
| :-------: | :---------------: |
|    5x5    |        25         |
|  2x(3x3)  |        18         |
|    7x7    |        49         |
|  3x(3x3)  |        27         |



![](https://cdn-images-1.medium.com/max/1600/1*Fw-ehcNBR9byHtho-Rxbtw.gif)

​							**Fig 1.5 :** 3x3 Convolution

_(image source: Irhum Shafkat from towardsdatascience.com)_

___

## Feature Maps

​	When we convolve Kernels/Filters over images we get to see features these number of features are termed as feature maps. This is very much required as we can visualize what's happening in the network after a lot of epochs and what features are created at what levels and which are the patterns that's taking prominence.

​	These feature maps will contain small features likes lines, blobs, gradients at lower level and will form closer to the class images for which we are training data to classify for as we proceed forward in the network.

​	![alt text](https://m2dsupsdlclass.github.io/lectures-labs/slides/04_conv_nets/images/lecunconv.png)

​					**Fig 1.6 :** Feature maps of a CNN at different levels

## Feature Engineering (older computer vision concept)

​	Before the popularization of Deep Learning, Computer Vision object detection the feature were written and chosen by the Computer vision engineers. But with the power of backpropagation these were made obsolete in fact hype of Deep Learning was because the programs could do these things based on data rather than manually creating and determining which filter to apply for what images.

​	This process required extensive domain specific knowledge and the data set should have to pre-processed a lot than the current practices also it had human manual intervention for selecting the features where it also accounted for error and unknown features which may had helped for a specific problem.

![](http://www.machinelearningtutorial.net/wp-content/uploads/2017/06/machine-learning-pipeline.png)

​						**Fig 1.7 :** Feature engineering in old pipeline

(_image source : machinelearningtutorial.net_)

___

## Activation Function

​	There are two types of activation function linear and non-linear functions, where linear functions is not some discrete function but the practice linear regression but this is only for data which can separated by a linear decision boundary. Non-linear functions on the other hand help separates different types of data which are difficult to separate linearly.

​	Some of the non-linear functions that are used are sigmoid, tanh, relu, leaky relu though sigmoid was popular and was efficient relu activation function took over it and was more effective.

​	In neural as non-linear functions are used for each node it'll form a more complex and suitable function as whole to optimally fit the data.

![](https://cdn-images-1.medium.com/max/1000/1*ZafDv3VUm60Eh10OeJu1vw.png)

​			**Fig 1.8 :** Some activation functions and their graphical representation

(_image source : Shruti Jadon from medium.com_)

___

## How to create an account on GitHub and upload a sample project

* Visit website https://github.com .

* The landing page will have the signup button click on it and fill the details in the signup form to create a account.

  ![](https://www.wikihow.com/images/thumb/2/2c/Join-github-1.jpg/728px-Join-github-1.jpg)

  ​								**Fig 1.9.1**

* Choose a plan :  

  - Public repositories for free 
  - 
  - Private repositories for a charge of $7/month

* Create a repository by following the steps below 

  * Click on new repository

  ![](https://help.github.com/assets/images/help/repository/repo-create.png)

  ​								**Fig 1.9.2**

  * Name your repository

  ![](https://help.github.com/assets/images/help/repository/create-repository-name.png)

  ​								**Fig 1.9.3**

  * Choose which type of repository you need

  ![](https://help.github.com/assets/images/help/repository/create-repository-public-private.png)

  ​								**Fig 1.9.4**

  * Tick on Initialize read me file so that u don't need to go through the hassle of adding the first file to your repo

  ![](https://help.github.com/assets/images/help/repository/create-repository-init-readme.png)

  ​								**Fig 1.9.5**

* Uploading project to your repo and enter comments for your change/upload.

  ![](https://everytoknow.com/wp-content/uploads/2018/09/Upload-files-%C2%B7-dinesh12jain-somthing.png)

  ​								**Fig 1.9.6**

___

## Receptive Field

​	Incase of data input being images its impractical to connect all the neurons like a fully connected Artificial Neural Network, that is why CNN are used so as to proceed with only the prominent features of the images by convolving and max pooling with reduction of spatial and depth dimensions without much loss of required data.

​	While we do the above mentioned steps each layer has the vision of previous layers and the dimensions differ but a layer can have the view of past n layers through features maps which are carrying the required data this connectivity is know as Receptive field.

​	

![](https://cdn-images-1.medium.com/max/1000/1*YpXrr8bN5XyqOlztKPHvDw@2x.png)

​								**Fig 1.10 :** Receptive of each pixel

(_image source : Arden Dertat from towardsdatascience.com_)
