# TensorBoard Tutorial

<p align="center">
<img src="../co_logo.png" alt="Clusterone" width="200">
</p>

This is a tutorial on how to use [TensorBoard](https://github.com/tensorflow/tensorboard), TensorFlow's visualization suite. The code uses [TensorFlow](https://tensorflow.org) and the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

The tutorial is separated into parts. Part 1 introduces graphs, scalar plots, and histograms. Part 2 focuses on outputting images to TensorBoard.

These links get you to the tutorials on the Clusterone blog:

- [Part 1: Graphs, Scalars, and Histograms](https://clusterone.com/tutorials/tensorboard-part-1)
- [Part 2: Images](https://clusterone.com/tutorials/tensorboard-part-2)

Follow the instructions below to run the tutorial code locally and on Clusterone. 

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [More Info](#more-info)
- [License](#license)

## Install

To run the code, you need:

- [Python](https://python.org/) 3.5
- [Git](https://git-scm.com/)
- TensorFlow 1.5+. Install it like this: `pip install tensorflow==1.7.0` (although the code may run with later versions, the code was written and tested using TF 1.7)
- The Clusterone Python library. Install it with `pip install clusterone`
- Clusterone account (to run on Clusterone). Create an account for free on [https://clusterone.com/](https://clusterone.com/).

For part 2 of the tutorial, you also need the following Python packages:
- [OpenCV](https://opencv.org/). Install it with `pip install opencv-python`

### Setting Up

#### To run locally
All you need to do is to clone this repository onto your local machine:

```shell
git clone https://github.com/clusterone/clusterone-tutorials
```

#### To run on Clusterone
Add a project by linking this GitHub repo (`clusterone/clusterone-tutorials`) as shown [here](https://docs.clusterone.com/documentation/projects-on-clusterone/github-projects#create-a-project-using-existing-github-repository).

## Usage

The tutorial code is divided into multiple stand-alone files.

Part 1:

- [main_bare.py](code/part_1/main_bare.py): A simple implementation of MNIST handwritten digit recognition. This file doesn't contain any visualization with TensorBoard and serves as the starting point for the tutorial.
- [main_tensorboard_graph.py](code/part_1/main_tensorboard_graph.py): Adds names and `tf.name_scope` to the code to produce a network graph in TensorBoard.
- [main_tensorboard.py](code/part_1/main_tensorboard.py): Extends main_tensorboard_graph to include scalar plots using `tf.summary`, as well as histograms.

Part 2:

- [main_tensorboard_images.py](code/part_2/main_tensorboard_images.py): Includes images of misclassified MNIST digits in TensorBoard.

You can run the code on your local machine, as well as on Clusterone without changing the code.

### Run the code locally

Navigate into the directory for [part 1](code/part_1/) or [part 2](code/part_2/) of the tutorial. Assuming all packages are installed correctly, you can run all script with `python <script-name>`. The script will print the necessary command to launch TensorBoard to the console.

### Run on Clusterone

These instructions use the `just` command line tool. It comes with the Clusterone Python library and is installed automatically with the library.

First, let's make sure that you have the `clusterone-tutorials` project. Execute the command `just get projects` to see all your projects. You should see something like this:
```shell
>> just get projects
All projects:

| # | Project                       | Created at          | Description |
|---|-------------------------------|---------------------|-------------|
| 0 | username/clusterone-tutorials | 2018-11-26T14:05:12 |             |
```
where `username` should be your Clusterone account name.

Let's create a job. Below is an example. Replace the path to python script and `requirements.txt` to the ones you want to run.

```shell
just create job single \
  --project clusterone-tutorials \
  --name tb-job \
  --command "python tensorboard/code/part_1/main_tensorboard.py" \
  --setup-command "pip install -r tensorboard/code/part_1/requirements.txt" \
  --docker-image tensorflow-1.8.0-cpu-py36 \
  --instance-type aws-t2-small
```

Now all that's left to do is starting the job:

```shell
just start job -p tensorboard/tb-job
```

That's it! You can monitor its progress on the command line using `just get events`. More elaborate monitoring is available on the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## More Info

To learn more about this tutorial, take a look at the corresponding articles on our [tutorial](https://clusterone.com/tutorials) page!

For further info on the MNIST dataset, check out [Yann LeCun's page](http://yann.lecun.com/exdb/mnist/) about it. To learn more about TensorFlow and Deep Learning in general, take a look at the [TensorFlow](https://tensorflow.org) website.

## License

[MIT](LICENSE) © Clusterone Inc.

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset has been created and curated by Corinna Cortes, Christopher J.C. Burges, and Yann LeCun.
