<p align='center'><a href='https://www.eventbrite.com/e/algorithmic-trading-with-python-cohort-2-tickets-1833367644979?aff=oddtdtcreator'><img src='https://static.packt-cdn.com/assets/images/packt+events/Python_for_Algorithmic Trading_v1.png'/></a></p>




# Applied Deep Learning and Computer Vision for Self-Driving Cars
<a href= "https://www.packtpub.com/in/data/hands-on-self-driving-cars-with-deep-learning?utm_source=github&utm_medium=repository&utm_campaign=9781838646301" /><img src="https://www.packtpub.com/media/catalog/product/cache/c2dd93b9130e9fabaf187d1326a880fc/9/7/9781838646301-original_42.jpeg" alt="Applied Deep Learning and Computer Vision for Self-Driving Cars" height="256px" align="right"></a>

This is the code repository for [Applied Deep Learning and Computer Vision for Self-Driving Cars](https://www.packtpub.com/in/data/hands-on-self-driving-cars-with-deep-learning?utm_source=github&utm_medium=repository&utm_campaign=9781838646301), published by Packt.

**Build autonomous vehicles using deep neural networks and behavior-cloning techniques** 

## What is this book about?

This book covers the following exciting features: 
* Implement deep neural network from scratch using the Keras library
* Understand the importance of deep learning in self-driving cars
* Get to grips with feature extraction techniques in image processing using the OpenCV library
* Design a software pipeline that detects lane lines in videos
* Implement a convolutional neural network (CNN) image classifier for traffic signal signs

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1838646302) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Errata

**Page 22**

It is: RADAR has great resolution;

Should be: LIDAR has great resolution;

It is: Let's look at the following RADAR chart:

Should be: Let's look at the following LIDAR chart:

It is: RFig 1.5: RADAR chart – strength

Should be: Fig 1.5: LIDAR chart – strength

**Page 23**

It is: Fig 1.6: RADAR and camera plot

Should be: Fig 1.6: LIDAR and camera plot

**Page 30**

It is: One of the shortcomings of LIDAR is that it usually has a low resolution;

Should be: One of the shortcomings of RADAR is that it usually has a low resolution;



## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

In the book you will also learn important python libraries like KERAS for Deep Learning and OpenCV for computer vision in detail. All codes are tested on latest Anaconda environment (https://www.anaconda.com/products/individual) with python 3.7 on Windows 16 GB laptop. It is recommended to use laptop with more than 8GB. You can also use Google Colab if you would like to execute the code in platform independent environment. You need to install below libraries:

1. Tensorflow Installation:  https://www.tensorflow.org/install
2. Keras Installation:  https://keras.io/
3. Pandas Installation:  https://pandas.pydata.org/
4. Numpy Installation:  https://numpy.org/
5. OpenCV Installation:  https://stackoverflow.com/questions/51853018/how-do-i-install-opencv-using-pip/56315658

**Note:** You can directly install Anaconda environment as it will install most of the datascience packages at once: https://www.anaconda.com/products/individual

Any additional installation instructions and information the user needs for getting set up. 

For few of the chapter you need to install few files and put in folder provides, please find the details below:

Chapter 3 to 6: No download required

Chapter 7: Download link is provided in folder “traffic-signs-data”, download the file and put in “traffic-signs-data”

Chapter 8-9: No Download required

Chapter 10: download link is provided in “beta_simulator_windows” and “track” folder. Download the files and put in these folders.

Chapter 11: download link can be found in “data” folder.


**Following is what you need for this book:**
If you are a deep learning engineer, AI researcher, or anyone looking to implement deep learning and computer vision techniques to build self-driving blueprint solutions, this book is for you. Anyone who wants to learn how various automotive-related algorithms are built, will also find this book useful. Python programming experience, along with a basic understanding of deep learning, is necessary to get the most of this book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-12).

### Software and Hardware List

| Chapter  | Software required                                             | OS required                        |
| -------- | ------------------------------------                          | -----------------------------------|
| 1        |Install Python 3.7 with latest Anaconda environment            | Windows, Mac OS X, and Linux (Any) |
| 2        |Install deep learning libraries TensorFlow 2.0 and KERAS 2.3.4 | Windows, Mac OS X, and Linux (Any) |
| 3        |Install image processing library OpenCV                        | Windows, Mac OS X, and Linux (Any) |



We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [ https://static.packt-cdn.com/downloads/9781838646301_ColorImages.pdf] 

### Related products <Other books you may enjoy>
* PyTorch Computer Vision Cookbook [[Packt]](https://www.packtpub.com/in/data/pytorch-computer-vision-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781838644833) [[Amazon]] (https://www.amazon.com/dp/1838644830)

* Mastering Computer Vision with TensorFlow 2.x [[Packt]](https://www.packtpub.com/in/data/advanced-computer-vision-with-tensorflow-2-x?utm_source=github&utm_medium=repository&utm_campaign=9781838827069) [[Amazon]] (https://www.amazon.com/dp/1838827064)

## Get to Know the Authors
**Sumit Ranjan**
is a silver medalist in his Bachelor of Technology (Electronics and Telecommunication) degree. He is a passionate data scientist who has worked on solving business problems to build an unparalleled customer experience across domains such as, automobile, healthcare, semi-conductor, cloud-virtualization, and insurance.

He is experienced in building applied machine learning, computer vision, and deep learning solutions, to meet real-world needs. He was awarded Autonomous Self-Driving Car Scholar by KPIT Technologies. He has also worked on multiple research projects at Mercedes Benz Research and Development. Apart from work, his hobbies are traveling and exploring new places, wildlife photography, and blogging.

**Dr. S. Senthamilarasu**
was born and raised in the Coimbatore, Tamil Nadu. He is a technologist, designer, speaker, storyteller, journal reviewer educator, and researcher. He loves to learn new technologies and solves real world problems in the IT industry. He has published various journals and research papers and has presented at various international conferences. His research areas include data mining, image processing, and neural network.

He loves reading Tamil novels and involves himself in social activities. He has also received silver medals in international exhibitions for his research products for children with an autism disorder. He currently lives in Bangalore and is working closely with lead clients.

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.





### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781838646301">https://packt.link/free-ebook/9781838646301 </a> </p>