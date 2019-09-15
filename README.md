# E2EP-CL (End to End Processing - Command Line)

E2EP is a command line tool built in python that allows for the easy implementation of pre-processing algorithms and both a simple neural network and a convolutional neural network. Using E2EP, the user has a quick and easy way to play around, test a hypothesis, and make predictions without having to write code.

## Setup

Run the following line to install the tool
```
pip install E2EP
```
 
Run

```
pip install numpy scipy scikit-learn pyparsing matplotlib click tensorflow keras os math
```

to install every required library

## Process

E2EP is has two parts: *pre-processing* followed by *analysis*. Each step includes data visualization.

### Pre-Processing

Pre-processing reduces the dimensionality or the number of features or variables in the data so that the important features remain (to prevent overfitting).

Two pre-processing algorithms: Principal Component Analysis (PCA) and Bandpass.

PCA is used for general data. Bandpass is used for image data

#### PCA

Takes all of the features and converts them in a set of linearly uncorrelated variables

#### Bandpass

Bandpass filtering is used to downscale images (make more blurry) to improve computational efficiency

### Analysis

#### Neural Network (NN)

Trains simple neural network for general data and outputs the weights

**parameters:**

| name | type | description                                                                      | example        |
|------|------|----------------------------------------------------------------------------------|----------------|
| X    | file | a file path to a CSV which holds your training data                              | ./train.csv    |
| Y    | file | a file path to a CSV which holds your expected outputs for the training examples | ./expected.csv |

**flags:**

| name        | type   | description                                                     | default | example      |
|-------------|--------|-----------------------------------------------------------------|---------|--------------|
| --lam       | float  | The regularization amount                                       | 1       | 0.07         |
| --maxiter   | int    | The maximum iterations for chosen to minimise the cost function | 250     | 30           |
| --output    | string | A file path to save the minimised parameters to                 | nil     | ./output.csv |
| --normalize | bool   | Perform normalization on the training set                       | true    | false        |
| --verbose   | bool   | Output the training progress                                    | true    | false        |

**example:**

```
$ E2EP train ./X.csv ./Y.csv --output=./weights.csv --normalize=true
```

Once you run the train command the neural network will intialize and begin to learn the weights, you should see an output similar to bellow if the `--verbose` flag is set to true.



#### Convolutional Neural Network (CNN)







### Predict

The prediction command takes a set of learned weights and a given input to predict a an ouput. The learned weights are loaded into the neural network by providing an file which holds them in a rolled 1 * n vector shape. In order for the predict command to work correctly these parameters need to be unrolled and therefore you need to provide the sizes of the input layer, hidden layer, and output labels that you wish to unroll the 

**parameters:**

| name   | type | description                                                                        | example     |
|--------|------|------------------------------------------------------------------------------------|-------------|
| x      | file | the file that holds the 1 * n row example that should be predicted                 | ./input.csv |
| params | file | The file that holds a 1 * n rolled parameter vector (saved from the train command) | ./ouput.csv |
| labels | int  | The size of the output layer that the parameters were trained on                   | 3           |

**flags:**

| name        | type   | description                                                     | default | example      |
|-------------|--------|-----------------------------------------------------------------|---------|--------------|
| --normalize | bool   | Perform normalization on the training set                       | true    | false        |
| --sizeh     | int    | The size of the hidden layer if it differs from the input layer | nil     | 8            |

**example:**

```
$ neuralcli predict ./x.csv 3 ./params.csv 
```

Neuralcli will now print a prediction in INT form, corresponding to the index of you output labels.
e.g. `0` will correspond to you first classification label. 

### Test

The test command gives some primitive feedback about the correctness of your hypothesis by running a diagnostic check on the given data set and expected output. This method plots the the margin of prediction error against the increase in size of training examples. This can be useful to determine what is going wrong with your hypothesis, i.e. whether it is underfitting or overfitting the training set.

**parameters:**

| name | type | description                                                                      | example        |
|------|------|----------------------------------------------------------------------------------|----------------|
| X    | file | a file path to a CSV which holds your training data                              | ./train.csv    |
| Y    | file | a file path to a CSV which holds your expected outputs for the training examples | ./expected.csv |

**flags:**

| name        | type   | description                                                     | default | example      |
|-------------|--------|-----------------------------------------------------------------|---------|--------------|
| --lam       | float  | The regularization amount                                       | 1       | 0.07         |
| --maxiter   | int    | The maximum iterations for chosen to minimise the cost function | 250     | 30           |
| --normalize | bool   | Perform normalization on the training set                       | true    | false        |
| --verbose   | bool   | Output the training progress                                    | true    | false        |
| --step      | int    | The increments that the training will increase the set by       | 10      | 100          |

**example:**

```
$ neuralcli train ./X.csv ./Y.csv --step=50 --normalize=true
```

Neural cli will then run the test sequence printing its progress as it increases the size of the training set.

![](http://i.imgur.com/TFlhHJN.gif)

After this runs it will then print a plot of the hypothesis error against the size of training set the weights where learned on. Below is an example graph plotted from the iris dataset.

![](http://i.imgur.com/o3ZTQxY.png)
