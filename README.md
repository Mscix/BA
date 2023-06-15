# Active Learning+
This is the implementation for the research question: A Comparison Between the Use of Unlabeled and Weakly Labeled Data in Active Learning for a Text Classification Problem

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Usage](#usage)
4. [Unit Tests](#unit-tests)
5. [References](#references)

## Installation

Clone the repository to your local machine:

```shell
git clone https://github.com/Mscix/BA
```

Go inside the directory:

```shell
cd BA
```

Install requirements for this project:
```shell
pip3 install -r requirements.txt
```


## Data Preparation
Data set source: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset [Zhang et al., 2015].

Download the data set

Place it in the root of this project and run the following command

`````shell
python3 one_time_preprocessing.py
`````

Now you have all the subsets called: small.csv (40 instances), medium.csv (4000 instances), large.csv (40000) in the same folder as test.csv and train.csv


## Usage
### Weights and Biases Set-Up
[Set up a free Weights & Biases (wandb) account for convenient run tracking. It can be easily done by signing up with GitHub.](https://wandb.ai/login?signup=true)

If you are running this project in a notebook then run this to log in:
`````shell
import wandb
%env WANDB_API_KEY=our_API_key
wandb.login()
`````

If you are running this project locally through the command line (this will add the API key to you .netrc file. In case you want to log out
you will have to enter the file and remove the API key manually):
````shell
wandb login your_API_key
````
When you run the project it will be automatically tracked and the results will be logged to the active-learning-plus project.


### Hyperparameters

The following hyperparameters can be set for each run:

| Argument | Description                                                                                                                 | Input Type |
| --- |-----------------------------------------------------------------------------------------------------------------------------| --- |
| `-p` | Path to the training set, must be in CSV format                                                                             | String |
| `-tp` | Path to the test set, must be in CSV format                                                                                 | String |
| `-m` | Program mode: 'AL+' (Active Learning Plus) , 'AL' (Active Learning), 'Standard'                                             | String |
| `-sm` | Sampling method: 'Random', 'LC' (Least Confidence), 'EC' (Entropy), 'MC' (Margin of Confidence), 'RC' (Ratio of Confidence) | String |
| `-d` | Confidence delta, determines the pseudo-label acceptance threshold                                                          | Float, x ∈ [0,1] |
| `-ait` | The number of active learning iterations to perform                                                                         | Integer |
| `-ns` | Number of instances to select in each active learning iteration                                                             | Integer or Float |
| `-iss` | Number of instances to select for the initial step                                                                          | Integer or Float |
| `-err` | The error rate with which the Custom Labeler will label the training data                                                   | Float, x ∈ [0,1] |
| `-r` | This argument determines whether the model is reset each active learning (AL) iteration                                     | Boolean |
| `-pat` | This is the patience parameter for Early stopping                                                                           | Integer ≥ 0$ |

For additional help use the command
````shell
python3 main.py -h
````

## Unit Tests

To run the Unit Tests use: 
````shell
python3 -m unittest -v tests/test_file.py
````


## References
- Zhang, X., Zhao, J. J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. In NIPS.
