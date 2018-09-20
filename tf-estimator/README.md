# Distributed TensorFlow with Estimators

<p align="center">
<img src="../co_logo.png" alt="Clusterone" width="200">
<br>
<br>
<a href="https://slackin-altdyjrdgq.now.sh"><img src="https://slackin-altdyjrdgq.now.sh/badge.svg" alt="join us on slack"></a>
</p>

This is a tutorial on how to use TensorFlow's [Estimator class](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator), including creating an Estimator by importing a Keras model. The code uses the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

The tutorial itself is published on our [blog](https://clusterone.com/blog) and can be found [here](https://clusterone.com/blog/2018/09/19/distributed-tensorflow-estimator-class).

Follow the instructions below to run the tutorial code locally and on Clusterone. 

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [More Info](#more-info)
- [License](#license)

## Install

To run the code, you need:

- [Python](https://python.org/) 3.6
- [Git](https://git-scm.com/)
- TensorFlow 1.5. Install it like this: `pip install tensorflow`
- [NumPy](http://www.numpy.org/). Get it with `pip install numpy`
- The Clusterone Python library. Install it with `pip install clusterone`
- To run the code on Clusterone, you need a Clusterone account. [Join the waitlist](https://clusterone.com/join-waitlist/) here.


### Setting Up

Start out by cloning this repository onto your local machine. 

```shell
git clone https://github.com/clusterone/clusterone-tutorials
```

## Usage

You can run the tutorial code either on your local machine or on the Clusterone deep learning platform, even distributed over multiple GPUs. No code changes are necessary to switch between these modes.

### Run the code locally

Make sure you have all requirements installed that are listed above. Assuming all packages are installed correctly, you can run all script with `python mnist.py`. The script will download the mnist dataset and then start training.

### Run on Clusterone

These instructions use the `just` command line tool. It comes with the Clusterone Python library and is installed automatically with the library.

cd into the folder you cloned with `cd clusterone-tutorials/tf-estimator`  and log into your Clusterone account using `just login`.

First, create a new git repository in the code directory and commit the Python files to it:

```shell
git init
git add .
git commit -m "Initial commit"
```

Then, create a new project on Clusterone:

```shell
just init project tf-estimator
```

Now, upload the code to the new project:

```shell
git push clusterone master
```

Finally, create a job. Make sure to replace `YOUR_USERNAME` with your username.

```shell
just create job distributed --project YOUR_USERNAME/tf-estimator --module mnist --name first-job \
--time-limit 1h
```

This creates a job with 2 worker nodes and 1 parameter server. See our [documentation](https://docs.clusterone.com/cli-reference-documentation/just-create-job) for more information on how to change the number and instance types of worker and parameter servers.

Now all that's left to do is starting the job:

```shell
just start job -p tf-estimator/first-job
```

That's it! You can monitor its progress on the command line using `just get events`. More elaborate monitoring is available on the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## More Info

For further information on this example, take a look at the [tutorial](https://clusterone.com/blog/2018/09/19/distributed-tensorflow-estimator-class) based on this repository on the Clusterone Blog.

For further info on the MNIST dataset, check out [Yann LeCun's page](http://yann.lecun.com/exdb/mnist/) about it. To learn more about TensorFlow and Deep Learning in general, take a look at the [TensorFlow](https://tensorflow.org) website.

## License

[MIT](LICENSE) © Clusterone Inc.

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset has been created and curated by Corinna Cortes, Christopher J.C. Burges, and Yann LeCun.
