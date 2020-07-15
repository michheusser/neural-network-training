# Training of Handwritten Symbols 
As a part of a bigger task, the aim of this project is to train a neural network to understand and evaluate handwritten digits and a small array of mathematical operations. Currently, it definitely would not be hard to find existing libraries or tools within the community that would allow me to perform the same tasks described here. But the intention of this project was mainly to do everything from scratch, not only as a learning experience, but also as a way to demonstrate my way to approach the problem, my knowledge of machine learning, data manipulation, algorithms, data structures and a general approach to structuring and problem solving.

## Project overview
Upon having tried several existing databases for handwritten digits (e.g. MNIST) or similar ones that included mathematical symbols, that did not work well for the intended purpose, I decided to create my own training, validation and testing datasets from scratch which were then fed to the training algorithms. I developed two packages in Python to deal firstly with the image processing and data creation, and secondly with the creation and training of a feedforward neural network, with back-propagation and stochastic gradient descent.

### Image Processing and Data Manipulation Package (‘datatools’)
This package provides public high-level methods that allow the user to create lists of input-output-datapoints containing image information, and the desired symbolic representation. It allows the user to import images containing multiple hand-written instances of a certain symbol (e.g. a scanned image), and segment them to individual images of the desired proportions.

Thanks to the image processing tools, the package allows a user to artificially expand the dataset by performing batch-transformations (scaling / rotations) to each image, putting emphasis on the transformation of the handwritten strokes and not every pixel. A consequence of this, is that no matter how an image might be resized, strokes of one pixel widths will be scaled to the desired size, keeping the one pixel width. This is advantageous, because strokes won’t disappear when images are made smaller, nor will they become thicker when the images are made bigger. In the end, what is important when training a neural network, is that it understands the drawing stroke a user meant when writing the symbol, disregarding the tickness of the pen the user might have used. If one wanted to include information about the thickness, or add noise to the images, it would be easy, once a dataset is available that contains the stroke-information.

The package can be found in the following path:
```
neural-network-training/src/datatools
```
An example of the usage and API of this package can be found in the following file
```
neural-network-training/src/main_dataset_creation.py
```
 To download the dataset I created or see more details about its creation, please visit the kaggle repository:
https://www.kaggle.com/michelheusser/handwritten-digits-and-operators


### Neural Network Training Package (‘nntools’)
This package offers the tools to create, train, validate and test a simple feedforward neural network. It was made to work with the dataset format and classes created in the ‘datatools’ package (mentioned above). 

The main module is the ‘manipulator’ which provides a user-friendly API to create and train a neural network, using back-propagation to apply the stochastic gradient descent on the cost function. The cost function can be chosen to be mean-squared-error or cross-entropy with an included regularization parameter to help with overfitting.

An example of the usage and API of this package can be found in the following file
```
neural-network-training/src/main_neural_network_training.py
```

## Project Structure
```
.src
├── datatools                            # Data manipulation and image processing package
│   ├── __init__.py                      # Initialization of package
│   ├── dataset_processor.py             # API module to provide high-level functionalities
│   ├── io_datapoint.py                  # Datapoint module containing the input-output class to work with the neural network
│   └── Image_processing                 # Image processing sub-package
│      ├── __init__.py                   # Sub-package initialization
│      ├── image_data.py                 # Module containing the custom image-data object
│      ├── manipulator.py                # Module containing object for image manipulations transformations
│      └── segmentator.py                # Module containing object to segment images containing several symbols
│
├── nntools                              # Neural Network Training package
│   ├── __init__.py                      # Initialization of package
│   ├── manipulator.py                   # API module to provide high-level functionalities 
│   └── tools                            # Sub-package with specific modules
│      ├── __init__.py                   # Initialization of sub-package
│      ├── neural_network.py             # Module containing the neural-network class
│      ├── classifier.py                 # Module containing the object to classify outputs on a neural network
│      ├── validator.py                  # Module containing the object to validate a neural network on a dataset
│      └── trainer.py                    # Module containing the object to train a natural network on a dataset
│
├── main_dataset_creation.py             # Script containing steps for the creation of data
└── main_neural_network_training.py      # Script containing steps for the training of a neural network
```

