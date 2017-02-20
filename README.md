#**Behavioral Cloning** 

This project is part of Udacity's [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). The goal of the project is to build a Convolutional Neural Network (CNN) to clone human driving behaviour. 

Udacity provides a simulator based on Unity to use for training and testing the CNN. The simulator has two modes: Training and Autonomous mode. In Training mode, the car can be controlled by keyboard, mouse or a gamepad and the simulator records data to be used for training the CNN. In Autonomous mode, the car is controlled by the CNN which provides steering angles as an input to the simulator. Udacity provides additionally a python script to connect the simulator to the CNN.

The animated GIF images below show the devolped CNN driving the car autonomously on the 2 tracks offered by the simulator: 

Track 1            |  Track 2
:-------------------------:|:-------------------------:
![](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/track1_anim.gif)  |  ![](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/track2_anim.gif)

To successfully complete the project, one has to go through the following steps:

1) Use the simulator to collect data of good driving behavior.
2) Build a Convolution Neural Network in Keras that predicts steering angles from images.
3) Train and validate the model with a training and validation set.
4) Test that the model successfully drives around track one without leaving the road. The simulator provides two tracks to use for training and testing the model.

The above steps are covered in implementation details section below.


## File structure

The project includes the following files:

* **model.py**: This file contains the script to create and train the model
* **drive.py**: This file is used to connect the model to the simulator for driving the car in autonomous mode
* **model.h5**:  This file contains the trained convolution neural network (saved by Keras  after training)
* **README.md** This file describing the project.

## Running the model

To run the model, start the simulator in Autonomous mode and run the following command:

```sh
python drive.py model.h5
```
The script will connect to the simulator and provide it with steering angles.


## Implementation details

### Data collection

In the Training mode, the simulator captures images from three cameras mounted on the car: a center, right and left camera. In addition to the images, steering angle, throttle, brake and speed are also logged in a generated .csv file with the name "driving_log.csv". For model training, only images and steering angles are used.

Below is an example of captured images from the center, left and right cameras:

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Left_Right_Center_Examples.png)

A sample from "driving_log.csv" is shown below:

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/driving_log_sample.png) 

For the training, 3 laps were recorded driving around track one counter-clockwise direction and then the car was turned around and 3 other laps were recorded in clockwise direction. This helps overcome the fact that track one has a left turn bias and therefore generates a more balanced data-set.

This is how a histogram of steering angles would look like if we only drive counter-clockwise (zero angles excluded). Negative steering angles represent left turns.

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Counter_clockwise_driving.png)

A histogram of steering angles (zero angles excluded) in case of driving counter-clockwise looks as follows:

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Clockwise_driving.png)

Driving in both directions would result in a more balanced data set as shown in the histogram below (zero angles excluded):

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Driving_both_directions.png)




### Model architecture

The model used is an implementation of the paper [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) by Nvidia. The model consists of 5 convolutional layers and 3 fully connected layers. ReLU activation functions were used and the input images to the model were normalized by a Keras lambda layer. 
 
 The benefit of hard-coding the normalization in the model is that no further normalization is required during the inference phase in drive.py
 
I used Dropout in order to reduce overfitting and the model proved to generalize quite well when tested on track two which it has never seen before. 

Below is the model architecture from the console output of Keras:

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/model_arch.png)

### Data augmentation and preprocessing

The recorded data ended up being 6487 samples of which 80% is used for training (5189 samples), which is not enough for the model to generalize. As a result, data augmentation was used to extend the dataset. 

#### Using data from left and right cameras

The images from the left and right cameras are used to extend the dataset. Hence, for every sample, we obtain an extra two samples (Left and Right) by applying an offset to steering angle. 

    for index in range(0, len(train_data)):
       row = train_data[index]
       for col_index in range(0, 3):
           img_path = row[col_index]
           tokens = img_path.split("\\")
           virtual_path = tokens[-2] + '/' + tokens[-1]
           train_images.append(virtual_path)
       center_angle = row[3]
       train_angles.append(float(center_angle))
       train_angles.append(float(center_angle) + steering_correction)
       train_angles.append(float(center_angle) - steering_correction)

#### Brightness augmentation

Images brightness was augmented by conversion to YUV color space and then adding a random value to the Y-channel. An offset was used to prevent too dark generated images.

    def image_augment(image):
	    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
		img_yuv[:, :, 0] = img_yuv[:, :, 0] * (0.2 + np.random.uniform())
	    img_aug = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
	    return img_aug

An sample of brightness augmented images is shown below:

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/image_augmentation.png)
#### Image cropping

Images from the simulator are of size (160, 320, 3). I cropped the images to remove the hood of the car and the part above the horizon. The information contained in these parts are not relevant for training the model.

After cropping the image was resized to the size expected by the model (66×200×3).

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Uncropped.png)

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/cropped.png)

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Resized.png)

####  Horizontal and vertical shifts.

Horizontal and vertical shifts were applied to images to simulate the effect of the car at different positions in the lane. An offset was added to the steering angle to help train the model to recover to the center of the lane in case of deviating. Vertical shifts 
were added to simulate going downhill or uphill.

    def image_shift(image, steer):
	    tr_x = 100*np.random.uniform()-50
	    tr_y = 20*np.random.uniform()-10
	    translation_matrix = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
	    image_tr = cv2.warpAffine(image, translation_matrix, (img_cols, img_rows))
	    steer_ang = steer + tr_x * 0.01
	    return image_tr, steer_ang


####  Flipping images
Images were flipped and the steering angle reversed to further extend the dataset and reduce any steering bias.

![enter image description here](https://github.com/ahany/behavioral-cloning/blob/master/Readme_Images/Flipping_Example.png)

#### Python generator
A Python generator was used to generate data for training and validation for efficient memory consumption.

During training, the generator would work with local batches of half the size of the given batch. The generator then would yield two images out of every image after flipping.

The training generator crops each image, resize to the appropriate image size expected by the model, apply horizontal and vertical shifts, augment brightness and then yield the image along with its flipped version.


    def train_generator(train_img, train_ang, batch_size=128):
	    num_samples = len(train_img)
	    local_batch_size = int(batch_size / 2)
	    while True:
	        train_img, train_ang = shuffle(train_img, train_ang)
	        for offset in range(0, num_samples, local_batch_size):
	            batch_images = train_img[offset:offset + local_batch_size]
	            batch_angles = train_ang[offset:offset + local_batch_size]
	            images = []
	            angles = []
	            for image, angle in zip(batch_images, batch_angles):
	                raw_image = plt.imread(rootpath + '/' + image)
	                angle = float(angle)
	                cropped_image = raw_image[70:135, :]
	                resized_image = cv2.resize(cropped_image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
	                shifted_image, angle = image_shift(resized_image, angle)
	                augmented_image = image_augment(shifted_image)
	                flipped_image = cv2.flip(augmented_image, 1)
	                flipped_angle = -angle
	                images.append(augmented_image)
	                angles.append(angle)
	                images.append(flipped_image)
	                angles.append(flipped_angle)
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)

### Model training

The model was trained using Adam optimizer and mean squared error as a loss function. I used 20% of the training data for validation and 5 epochs for training.

For each epoch, the training generator yields 20000 training samples.

    model = get_nvidia_model()
	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_generator(train_images, train_angles, batch_size=128),
	                    samples_per_epoch=20000,
                        validation_data=val_generator(val_data, batch_size=128),
                        verbose=1,
                        nb_val_samples=len(val_data),
                        nb_epoch=5)

No test dataset was used and instead the model was directly tested in the simulator's autonomous mode.

### Results

The car manages to drive well on both tracks mostly in the center of the lane although track 2 was never seen by the model in the training or validation phase.

Although the simulator is a very simplified version of the real world but it helps give an idea about the capability of deep learning models.

