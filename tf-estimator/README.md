# Distributed TensorFlow with Estimators

<p align="center">
<img src="../co_logo.png" alt="Clusterone" width="200">
<br>

This is a tutorial on how to use TensorFlow's [Estimator class](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator), including creating an Estimator by importing a Keras model. The code uses the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

The tutorial itself is published on our [blog](https://clusterone.com/tutorials) and can be found [here](https://clusterone.com/tutorials/distributed-tensorflow-part-2).

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
- TensorFlow 1.5 or higher. Install it like this: `pip install tensorflow`
- The Clusterone Python library. Install it with `pip install clusterone`
- GitHub account. Create an account for free on [https://github.com/](https://github.com/)
- To run the code on Clusterone, you need a Clusterone account. [Join the waitlist](https://clusterone.com/join-waitlist/) here.


### Setting Up

Follow the **Set Up** section of the [Get Started](https://docs.clusterone.com/get-started#set-up) guide to add your GitHub personal access token to your Clusterone account.

Then follow [Create a project](https://docs.clusterone.com/get-started#create-a-project) section to add clusterone-tutorials project. Use **`clusterone/clusterone-tutorials`** repository instead of what is shown in the guide.

## Usage

You can run the tutorial code either on your local machine or on the Clusterone deep learning platform, even distributed over multiple GPUs. No code changes are necessary to switch between these modes.

### Run the code locally

Start out by cloning this repository onto your local machine. 

```shell
git clone https://github.com/clusterone/clusterone-tutorials
```

Then navigate to the directory with `cd clusterone-tutorials`.

Make sure you have all requirements installed that are listed above. Assuming all packages are installed correctly, you can run all script with `python mnist.py`. The script will download the mnist dataset and then start training. You can view the training results with Tensorboard with `tensorboard --logdir=logs`.

### Run on Clusterone

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

Let's create a job. Make sure to replace `username` with your username.
```shell
just create job distributed \
  --project username/clusterone-tutorials \
  --name distributed-mnist-job \
  --worker-replicas 2 \
  --worker-type aws-t2-small \
  --docker-image tensorflow-1.11.0-cpu-py35 \
  --ps-replicas 1 \
  --ps-type aws-t2-small \
  --ps-docker-image tensorflow-1.11.0-cpu-py35 \
  --time-limit 1h \
  --command "python tf-estimator/main.py" \
  --setup_command "pip install -r tf-estimator/requirements.txt"
```

This creates a job with 2 worker nodes and 1 parameter server. See our [documentation](https://docs.clusterone.com/cli-reference-documentation/just-create-job) for more information on how to change the number and instance types of worker and parameter servers.

Now the final step is to start the job:

```shell
just start job -p clusterone-tutorials/distributed-mnist-job
```

That's it! You can monitor its progress on the command line using `just get events`. More elaborate monitoring is available on the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## More Info

For further information on this example, take a look at the [tutorial](https://clusterone.com/blog/2018/09/19/distributed-tensorflow-estimator-class) based on this repository on the Clusterone Blog.

For a more updated MNIST example, check out our [MNIST repo](https://github.com/clusterone/mnist). We also have other examples [here](https://docs.clusterone.com/examples).

For further info on the MNIST dataset, check out [Yann LeCun's page](http://yann.lecun.com/exdb/mnist/) about it. To learn more about TensorFlow and Deep Learning in general, take a look at the [TensorFlow](https://tensorflow.org) website.

If you have any further questions, don't hesitate to write us a support ticket on [Clusterone.com](https://clusterone.com) or join us on [Slack](https://bit.ly/2OPc6JH)!

## License

[MIT](LICENSE) © Clusterone Inc.

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset has been created and curated by Corinna Cortes, Christopher J.C. Burges, and Yann LeCun.
