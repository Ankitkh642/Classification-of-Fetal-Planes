Classification of Maternal
fetal ultrasound planes using CNN

In this study, we have tried to use these techniques of AI and Deep Learning
(namely CNN) to demonstrate how it can be used in case of medical application
and can save a lot of time of experts, essentially saving a lot of lives. Because time
can be of the very essence in the field of medical, where every second utilised can
save lives.
The purpose of this study was to assess the maturity of existing Deep
Learning classification approaches for use in a real-world maternal-fetal
clinical setting. And also give better results than their human couterparts to
better access the situation of fetus.

1. Tools:

Deep Learning architecture with neural network are the used tools here. Artificial
neural networks are inspired by the human biological nervous system in terms of
structure and idea.
The preceptron was one of the first neural networks based on the human brain
system. It had a directly connected input and output layer and was useful for
classifying linearly separable patterns.
A layered architecture, consisting of an input layer, an output layer, and one or more hidden layers, was devised to solve increasingly complex patterns.
A neural network is made up of interconnected neurons that take in data, process
it, and then forward the output of the current layer to the next layer. Because
hidden layers record non-linear relationships, having more hidden layers allows
you to cope with more complex situations.
Deep neural networks are the name given to these neural networks.
Deep learning offers a new cost-effective way to train DNNs, which were
previously sluggish to learn the weights.
Extra layers in DNN allow for the compilation of features from lower layers to the upper layer, allowing for complicated data modelling.
A hierarchical feature representation is formed by a deep neural network that
stacks numerous layers of neurons in a hierarchical manner.
There are nearly 1,000 layers presently!
After successful training with a sufficiently big knowledge database, a deep
network can essentially memorise all possible mappings and make intelligent
predictions, such as interpolations and/or extrapolations for unseen scenarios,
thanks to its massive modelling capacity.
Deep learning is having a significant impact on computer vision and medical
imaging as a result of this.
In reality, in realms like text, speech, and others, a similar effect is occurring.
Convolutional neural networks, for example, are among the deep learning
methods used in research.

2. Methodology:

I. Data (Image) Transformation:
To carefully analyze and model the given image data, data transformation is
needed. In computer vision, Data Augmentation is very important to
regularize your network and increase the size of your training set. There are
many Data transformation techniques *(rotation, flip, crop, etc...) that can
change the images’ pixel values but still keep almost the whole information
of the image, so that human could hardly tell whether it was augmented or
not. This forces the model to be more flexible with the large variation of
object inside the image, regarding less on position, orientation, size, color,...
Models trained with data augmentation usually generalize better but the
remaining question is to what extend the Data Transformation can improve
the performance of CNN model on image dataset. Through this step, we can
even work with a relatively smaller set of data, because it gives us an
extended version of the dataset.
Both images are
similar to each
other, because
they are
obtained by
transformation.

II. Image Resize:
Because the standard algorithms like VGG16 and RESNET50 follow
standard image size, the size of all the images in the dataset was fixed to a
greyscale of [224 X 224].
This is the standard image size of many algorithms and it also decides the
number of neurons on the first layer. Since these algorithms have fixed and
pre-calculated weights and neurons, the image has to be fixed to match the
default size.
III. Image Superimposition: Most of the highly used algorithms are designed for
colored image, hence, they have the RGB scale. That means, the default
formation and size of image in these techniques is [224 X 224 X 3], 3
denoting the RGB scale.To counter this, the images we have had to be bent according to the input
required. Hence, we took 3 copies of grayscale images, and superimposed on
each other, thus making the last dimension 3.
IV. Weights trainable: To reduce the burden on the machine, and decrease the
time taken to train the model, the trainable weights had to be stopped. If this
step is skipped, the accuracy is improved drastically, but we would, need
high computation power to perform this, which is not feasible.
V. Model fitting and training: Finally, the model was trained by dividing the
above obtained data into ‘Training set’ and ‘Validation set’. Number of
epochs were modified to find a balance between the accuracy and time taken to train it.



    • Image size : [224 X 224 X 3].
    • Weights : Imagenet.
    • Activation Function : Softmax
    • Loss Function : Categorical Crossentropy.
    • Optimizer : Adam.
    • Metrics : Accuracy.
    • Number of Epochs: 50


3. Results:

Accuracies of models(%):


	1. VGG16: 98.91 and 84.97
	2. RESNET50: 81.34 and 76.74
	3. DENSENET169: 99.05 and 85.92

[For further analysis of losses and accuracy variations with epochs, go through graphs directory of this repository.]





Shortcomings and Future opportunities:

    • This model, while being robust, is also very streamlined. It is tailored to this specific dataset and as such, might not be very accurate in an environment where the data provided to it is very different and outlying and anomaly. That is why the expert opinion might be needed after all.  This is the problem of the Deep learning models overall. 
    • Since the data is limited, the model may have a little bias. Even specialists can have biases, but we would want our model to be as free of bias as possible.	
    • We recognise that many other methods might have been benchmarked, and that computational models could have benefited from the application of prior processes such as image segmentation rather than evaluating photos as a whole as the study's principal shortcoming (especially in the case of fine-grained brain plane recognition) .
    • Images have to be labelled by the experts first, which is a drawback, because the models in deep learning are mostly based on supervised learning. The need of a specialist can’t be eliminated thus despite these breakthroughs in advancements of AI and Deep Learning.
