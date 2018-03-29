# Titanic Tutorial

<p align="center">
<img src="../co_logo.png" alt="Clusterone" width="200">
<br>
<br>
<a href="https://slackin-altdyjrdgq.now.sh"><img src="https://slackin-altdyjrdgq.now.sh/badge.svg" alt="join us on slack"></a>
</p>

This tutorial teaches you how to use [TensorFlow](https://tensorflow.org) to predict the survival of passengers of the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) based on the passenger list of the vessel's tragic maiden voyage.

This repository contains the code and data files required to run the tutorial model. For the tutorial itself, please [see here](https://medium.com/clusterone/tensorflow-beginner-guide-titanic-dataset-clusterone-7c134e447f3c).

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [More Info](#more-info)
- [License](#license)

## Install

To run this project on your local machine, you need:

- [Python](https://python.org/) 3.5
- [Git](https://git-scm.com/)
- The TensorFlow Python library. Install it using `pip install tensorflow`
- The Clusterone Python library. Install it with `pip install clusterone==2.0.0a03`

Additionally, to run the code on the Clusterone platform, you need a Clusterone account. [Sign up](https://clusterone.com/) for free if you don't have an account yet.

To get ready to use the Titanic tutorial, clone this repository to your machine:

```shell
$ git clone https://github.com/clusterone/clusterone-tutorials
```
## Usage

`cd` into the code directory of the Titanic tutorial. The tutorial contains 2 Python script.

### Titanic Basic

The basic version of the Titanic tutorial trains the neural network using only the features `pclass` and `age`. It achieves an accuracy on the test set of approximately 70-75%.

You can run the script with:

```shell
python titanic_basic.py
```

### Titanic

The complete version of the tutorial adds sex, family members on board, and the port of embarkation as input parameters to the model. This raises the prediction accuracy on the test set to around 80%.

You can run this script by typing:

```shell
python titanic.py
```

### Running on Clusterone

To run the code on Clusterone, you have to create a project and a dataset on the platform. The following instruction use the `just` command line interface, which is automatically installed together with the Clusterone Python package.

Start by logging into your Clusterone account with `just login`.

In the code directory, create a new Clusterone project and upload the code to Clusterone:

```shell
git init
just init project titanic
git add .
git commit -m "Initial commit"
git push clusterone master
```

Now, `cd` into the data directory. Here, create a dataset an upload the data to Clusterone.

```shell
just create dataset titanic-data
git init
just ln dataset -p titanic-data
git add .
git commit -m "Initial commit"
git push clusterone master
```

With your dataset and project ready, you can run your code on Clusterone. This is done by creating and starting a job.

```shell
just create job --name titanic-job --project titanic --datasets titanic-data --module titanic.py
just start job -p titanic/titanic-job
```

To monitor your job, head to the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## License

[MIT](LICENSE) © ClusterOne Inc.

The Titanic dataset is freely available as part of the public domain.