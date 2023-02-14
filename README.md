Narimanov Nurdaulet, Zhumatay Aruzhan BDA-2101

Report

GitHub account: https://github.com/chronaas

YouTube Video: https://youtu.be/tnBl7xpDj4A

1. Introduction

1.1 Problem
Generative Adversarial Networks (GANs) are a type of deep learning model that are used for generating new data, such as images, videos, and audio. One of the main challenges with GANs is that they can be difficult to train and the generated data may not always be of high quality or suitable for the intended purpose.
For example, in a GAN for generating images, the generator network may generate blurry or low-resolution images that do not closely resemble the training data. The training process can be unstable and prone to failure, resulting in models that generate poorly or not at all.
Another challenge is that GANs are a type of black box model, meaning that it can be difficult to understand how they are generating data or to modify the generated data to meet specific requirements. This can make it challenging to use GANs in real-world applications where the quality and suitability of the generated data is important.
These challenges pose a significant problem for researchers and practitioners who want to use GANs for generating high-quality, usable data. To address these challenges, it is necessary to develop more robust and stable GAN models that can produce high-quality data, as well as new techniques for understanding and modifying GANs to meet specific requirements.

1.2 Literature Review (another solutions).
Stabilizing GAN Training. One of the main challenges with GAN training is that it can be unstable, with the generator and discriminator networks competing in ways that cause the training process to fail. To address this issue, researchers have proposed a variety of techniques for stabilizing the GAN training process, including the use of minibatch discrimination, feature matching, and Wasserstein GANs.
Improving GAN Image Quality. Another challenge with GANs is that they often generate low-quality or blurry images that do not closely resemble the training data. To address this issue, researchers have proposed a number of approaches for improving the quality of generated images, including the use of progressive growing, self-attention mechanisms, and cycle-consistent GANs.
Understanding and Modifying GANs: As mentioned earlier, GANs are a type of black box model, which can make it challenging to understand how they are generating data and to modify the generated data to meet specific requirements. To address this issue, researchers have proposed a number of techniques for understanding and modifying GANs, including using activation maximization, interpretable GANs, and generative interpretable networks.
GANs for Specific Applications: GANs have a wide range of potential applications, and researchers have been exploring the use of GANs for a variety of specific tasks, including generating synthetic data for machine learning, creating realistic simulations for gaming, and improving image and video compression.

1.3. Current work (description of the work).

1.3.1. Import Data.
The code starts by installing the following packages using the pip package manager:
•	tensorflow
•	tensorflow-gpu
•	matplotlib
•	tensorflow-datasets
•	ipywidgets
After the packages have been installed, the code imports the TensorFlow library and sets the memory growth for any GPUs available on the system to True. This will allow TensorFlow to dynamically allocate memory on the GPU as needed, instead of pre-allocating all of the GPU memory.
The code then imports the tensorflow_datasets package and uses it to load the fashion_mnist dataset. The fashion_mnist dataset is a set of 28x28 grayscale images of various clothing items, such as shoes, shirts, and pants. The dataset is split into a training set and a test set.
The code uses the as_numpy_iterator method to get a Python iterator over the data in the ds dataset, and it accesses the next item in the dataset using the next() function. This item is then accessed using the ['label'] key to get the label for the image. The label is an integer representing the class of the clothing item in the image.

1.3.2. Build Generator.
Function build_generator that creates a generator model using the Keras Sequential API. The generator model takes in a random vector with 128 dimensions and generates an image.
The generator model starts by adding a dense layer that takes in the random vector and reshapes it to a 7x7x128 tensor. This layer is followed by two leaky rectified linear unit (LeakyReLU) activation layers, which serve to add non-linearity to the model.
Next, the generator model performs two upsampling operations using the UpSampling2D layer, which increases the spatial dimensions of the feature map, followed by two convolutional layers (Conv2D) with 128 filters and a kernel size of 5 with padding set to 'same'. These layers help to increase the spatial resolution of the image.
The generator model then adds two more convolutional blocks, each with 128 filters and a kernel size of 4, followed by a final convolutional layer with 1 filter and a kernel size of 4. The final layer is followed by a sigmoid activation function, which ensures that the output of the generator model is a binary image with pixel values between 0 and 1.
The function returns the generator model, which is a Keras Sequential model composed of the layers described above.
Code generates 4 new fashion items using the generator model. The input to the generator model is a random noise vector with shape (4, 128, 1). The predict method is then called on the generator model with the random noise as the input. This generates the new fashion images.
Next, a subplot with 4 columns is created using fig, ax = plt.subplots(ncols=4, figsize=(20,20)). The loop then iterates over the generated images and plots each image on a separate subplot using ax[idx].imshow(np.squeeze(img)). The plot title is set as the index of the image using ax[idx].title.set_text(idx).

1.3.3. Build Discriminator.
This code defines a function build_discriminator() which creates a Sequential model for a discriminator network in a Generative Adversarial Network (GAN).
The model consists of several Conv2D layers with a filter size of 5 and increasing numbers of filters, followed by LeakyReLU activation functions with a negative slope of 0.2 and Dropout layers with a rate of 0.4. These layers are used to extract features from the input images.
The output of the last Conv2D layer is flattened and passed to a Dropout layer and a dense layer with a single output unit and a sigmoid activation function. The sigmoid activation function is used to ensure the output of the model is between 0 and 1, and can be interpreted as the probability that the input image is real.
This discriminator model is used in a GAN to distinguish real images from generated images, with the goal of training the generator to produce images that are indistinguishable from real ones.

1.3.4. Construct Training Loop.

1.3.4.1. Setup Losses and Optimizers.
This code imports the Adam optimizer and the Binary Crossentropy loss function from Tensorflow's keras library. Then, it sets two instances of the Adam optimizer, g_opt and d_opt, with different learning rates for the generator and discriminator, respectively. Two instances of the Binary Crossentropy loss function, g_loss and d_loss, are created for the generator and discriminator, respectively. The learning rate is a hyperparameter that controls the step size at which the optimizer makes updates to the model parameters in order to minimize the loss function. The learning rate is set to 0.0001 for the generator optimizer and 0.00001 for the discriminator optimizer in this code.

1.3.4.2. Build Subclassed Model.
The FashionGAN class is derived from the Model class in TensorFlow, which provides a high-level API for building and training models. The FashionGAN class takes two arguments in its constructor: a generator model and a discriminator model. These two models are stored as attributes of the FashionGAN object.
The compile method takes several arguments including optimizers for the generator and discriminator models, as well as losses for each model. These optimizers and losses are also stored as attributes of the FashionGAN object.
The train_step method implements a single step of the GAN training process. It first generates a batch of fake images using the generator model. The discriminator model is then trained on both real and fake images. The labels for the real and fake images are created and then added with some random noise to make the training more robust. The binary cross-entropy loss between the predicted labels and the actual labels is then calculated and used to update the weights of the discriminator model using backpropagation.
Next, the generator model is trained by generating new images and using the discriminator model to make predictions on these images. The loss is calculated between the predicted labels and a vector of zeros, and this loss is used to update the weights of the generator model using backpropagation.
Finally, the method returns a dictionary of losses for the generator and discriminator models, which can be used for monitoring the progress of the training.

1.3.4.3. Build Callback.
The ModelMonitor class takes two arguments: num_img and latent_dim. num_img is the number of images to generate after each epoch and latent_dim is the dimension of the latent space in which the generator produces images.
The class implements the on_epoch_end method, which is called by Tensorflow Keras at the end of each epoch. In this method, random_latent_vectors are generated using tf.random.uniform method. These random latent vectors are then passed through the generator model to produce generated_images. The pixel values in the generated images are then scaled to the range [0, 255].
Finally, the generated images are saved to the images folder, with the name format generated_img_<epoch number>_<image number>.png. This allows the user to see the progression of the generated images as the model trains.

1.3.4.4. Review Performance.
Generate a plot to visualize the loss history of a GAN model. The plot has a title "Loss" and two lines, one for the discriminator loss and one for the generator loss. The hist object contains the training history of the GAN model, including the values for the discriminator loss (hist.history['d_loss']) and the generator loss (hist.history['g_loss']). These values are plotted using the plot function from the matplotlib.pyplot library, which is imported as plt. The legend function adds a legend to the plot to indicate which line corresponds to which loss. Finally, the show function displays the plot on the screen.
  
1.4 Generate Images.
The generator model is loaded using the "load_weights" method and the path to the model's weights file, which is stored in the "archive" directory and named "generatormodel.h5".
Once the generator model is loaded, it is used to generate images by calling the "predict" method and passing in a tensor of random normal values with shape (16, 128, 1) as input. The output of the generator model is then stored in the "imgs" variable.
Next, a subplot with 4 rows and 4 columns is created using the "subplots" function from the "matplotlib.pyplot" library. The created subplot is stored in the "fig" and "ax" variables.
Finally, a loop is used to populate each of the 16 subplots with one of the generated images. The image is plotted using the "imshow" method and the corresponding image data is taken from the "imgs" variable.


2. Data and Methods.
  
2.1. Information about the data.
We use the numpy and matplotlib libraries to import and display a sample of data stored in the dataiterator object. The code sets up a connection to the data source and retrieves the 'image' field of the first sample in the data set using dataiterator.next()['image'].
Then creates a subplot formatting with 4 columns using the fig and ax objects, and sets the size of each plot to be 20x20. In the for loop, the code grabs 4 samples from the data set and plots each sample as an image in the subplot using imshow(np.squeeze(sample['image'])). Finally, the label of each sample is added as the plot title using ax[idx].title.set_text(sample['label']).

2.2.  Description of the ML models you used with some theory
A generative adversarial network (GAN) is a deep learning architecture consisting of two components, a generator and a discriminator, that compete against each other in a zero-sum game. The generator tries to produce synthetic data that is similar to the real data, while the discriminator tries to determine whether a given data sample is real or synthetic. Over time, the generator and discriminator improve until the generator produces data that is indistinguishable from the real data.
The generator in a GAN is a neural network that takes a random noise vector as input and produces a synthetic sample as output. The synthetic sample is then fed into the discriminator, along with real data, to determine whether it is real or synthetic. The generator is trained to maximize the likelihood that the discriminator will classify its output as real, while the discriminator is trained to correctly classify the input data as real or synthetic.
The discriminator in a GAN is also a neural network that takes an input data sample and produces a scalar output indicating the likelihood that the input is real. The discriminator is trained to maximize this output for real data and minimize it for synthetic data produced by the generator.

3. Results.
  
![Рисунок1](https://user-images.githubusercontent.com/97881086/218848443-aa9ca26b-8808-465a-b05d-7b0f279fd862.png)
![image](https://user-images.githubusercontent.com/97881086/218848651-5cecd2fb-8694-4f4d-9fba-bcca5baff7a0.png)
![image](https://user-images.githubusercontent.com/97881086/218848684-c1b1268d-1757-43bb-be70-a04dc86e9da2.png)
![image](https://user-images.githubusercontent.com/97881086/218848708-dea35dce-6dac-4998-92c1-9859a0f43f2c.png)
![image](https://user-images.githubusercontent.com/97881086/218848730-635fde3a-80b6-402f-a407-68c4eae0c995.png)
![image](https://user-images.githubusercontent.com/97881086/218848752-174fb0ab-e877-4395-9122-03a6338a8e7a.png)
![image](https://user-images.githubusercontent.com/97881086/218848795-483ef3b7-75a1-4594-b342-7484f7d0b523.png)
![image](https://user-images.githubusercontent.com/97881086/218848809-606b271d-2487-4941-b61f-b854ed94a34c.png)

4. Discussion.
  
4.1. Critical review of results.
Results provides an implementation of a Generative Adversarial Network (GAN) trained on the Fashion MNIST dataset. The GAN consists of two parts, a generator and a discriminator, both of which are implemented as Convolutional Neural Networks (ConvNets). The generator is trained to generate images that look like fashion items, and the discriminator is trained to identify whether an image is real or fake. The two networks are trained together, with the generator trying to create images that fool the discriminator and the discriminator trying to identify real images.


4.2. Next steps.
Based on the project that we provided, we have implemented a basic GAN using the TensorFlow and TensorFlow Datasets libraries for the Fashion MNIST dataset. We would like to continue working on this project and provided suggestions for next steps:
Improve the model performance. Consider experimenting with different hyperparameters, such as the number of filters in the convolutional layers, the number of neurons in the dense layers, the activation functions, and the learning rates.
Use a larger dataset. The Fashion MNIST dataset is relatively small, with only 60,000 training samples. We could try training GAN on a larger dataset to see if it can generate higher-quality images.
Try different architectures. There are many different GAN architectures that have been proposed, and we could try implementing some of them to see how they compare to the architecture that we have used.
Evaluate the quality of generated images. The project only generates a few random images, and it is not clear how to quantify the quality of the generated images. We could try using existing metrics to evaluate the quality of the generated images.


References:
Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. https://arxiv.org/abs/1606.03498 
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. https://arxiv.org/abs/1710.10196 
Zhang, H., Xu, T., Li, H., Zhang, S., Huang, X., Wang, X., & Yang, Q. (2018). Self-attention generative adversarial networks. https://arxiv.org/abs/1805.08318 





