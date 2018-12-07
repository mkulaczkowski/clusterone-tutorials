# Titanic Tutorial

<p align="center">
<img src="../co_logo.png" alt="Clusterone" width="200">
</p>

This tutorial teaches you how to use [TensorFlow](https://tensorflow.org) to predict the survival of passengers of the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) based on the passenger list of the vessel's tragic maiden voyage.

This repository contains the code and data files required to run the tutorial model. For the tutorial itself, please [see here](https://clusterone.com/tutorials/tensorflow-titanics).

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Install

To run this project on your local machine, you need:

- [Python](https://python.org/) 3.5
- [Git](https://git-scm.com/)
- The TensorFlow Python library. Install it using `pip install tensorflow`
- The Clusterone Python library. Install it with `pip install clusterone`

Clone this repository to your machine:
```shell
$ git clone https://github.com/clusterone/clusterone-tutorials
```

To run this project on Clusterone, you need:
- Clusterone account. Create a free account on [https://clusterone.com/](https://clusterone.com/).

That's all you need! Add a project by linking this GitHub repo (`clusterone/clusterone-tutorials`) as shown [here](https://docs.clusterone.com/documentation/projects-on-clusterone/github-projects#create-a-project-using-existing-github-repository).

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

These instructions use the `just` command line tool. It comes with the Clusterone Python library and is installed automatically with the library.

If you have used Clusterone library before with a different Clusterone installation, make sure it is connected to the correct endpoint by running `just config endpoint https://clusterone.com`.

Log into your Clusterone account using `just login`, and entering your login information.

First, let's make sure that you have the project. Execute the command `just get projects` to see all your projects. You should see something like this:
```shell
>> just get projects
All projects:

| # | Project                       | Created at          | Description |
|---|-------------------------------|---------------------|-------------|
| 0 | username/clusterone-tutorials | 2018-11-20T00:00:00 |             |
```
where `username` should be your Clusterone account name.

With your project ready, you can run your code on Clusterone. This is done by creating and starting a job. Make sure to replace `username` with your username.

```shell
just create job --name titanic-job --project clusterone-tutorials --module titanic/code/titanic.py
just start job -p clusterone-tutorials/titanic-job
```

To monitor your job, head to the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## License

[MIT](LICENSE) © ClusterOne Inc.

The Titanic dataset is freely available as part of the public domain.
