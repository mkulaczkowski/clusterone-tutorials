# Horovod Tutorial

<p align="center">
<img src="../../co_logo.png" alt="Clusterone" width="200">
</p>

This tutorial presents an example to learn how to use [Horovod](https://github.com/uber/horovod).

This repository contains the code required to run the tutorial model. For the tutorial itself, please [see here](https://clusterone.com/tutorial/openmpi-introduction).

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [More Info](#more-info)
- [License](#license)

## Install

### Prerequisites
To run this project on your local machine, you need:

- [Python](https://python.org/) 3.5
- [Git](https://git-scm.com/)
- The TensorFlow Python library. Install it using `pip install tensorflow`
- The Horovod Python library. Install it using `pip install horovod`
- The Clusterone Python library. Install it with `pip install clusterone`

If you're adventurous and want to test MPI locally, I suggest pulling a pre-built Docker image. For example, `docker pull uber/horovod:0.15.0-tf1.11.0-torch0.4.1-py3.5` (be warned, the image is 3GB--see more options [here](https://hub.docker.com/r/uber/horovod/tags/))

To run this project on Clusterone, you need:
- GitHub account. Create a free account on [https://github.com/](https://github.com/).
- Clusterone account. Create a free account on [https://clusterone.com/](https://clusterone.com/).

Make sure you've added your [GitHub access token](https://docs.clusterone.com/account/third-party-apps/github-account) to your account.

Then link this repo (`clusterone/clusterone-tutorials`) as a project like shown [here](https://docs.clusterone.com/documentation/projects-on-clusterone/github-projects).

## Usage

You can run this code either locally or on Clusterone platform without any code changes. You can also run this in single instance or distributed mode without any code changes.

### Local

Start out by cloning this repository onto your local machine.

```shell
git clone https://github.com/clusterone/clusterone-tutorials
```

Then cd into the xgboost folder with `cd clusterone-tutorials/openmpi/horovod`.

Single instance mode is very simple. You just execute main.py:
```shell
python main.py
```

For distributed mode, you need to have a working MPI. If you have Docker, you can do this:
```shell
docker run -it -v $(pwd):/code uber/horovod:0.15.0-tf1.11.0-torch0.4.1-py3.5 /bin/bash
uber/horovod> cd /code
uber/horovod> mpirun --allow-run-as-root -np 3 -H localhost:3 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib python main.py
```
This will start three local processes running synchronous distributed training.

### Clusterone

These instructions use the `just` command line tool. It comes with the Clusterone Python library and is installed automatically with the library.

If you have used Clusterone library before, make sure it is connected to the correct endpoint by running `just config endpoint https://clusterone.com`,
then log into your Clusterone account using `just login`.

First, let's make sure that you have the `clusterone-tutorials` project. Execute the command `just get projects` to see all your projects. You should see something like this:
```shell
>> just get projects
All projects:

| # | Project                       | Created at          | Description |
|---|-------------------------------|---------------------|-------------|
| 0 | username/clusterone-tutorials | 2018-11-29T01:50:23 |             |
```
where `username` should be your Clusterone account name. If you don't have this project, see [Install](#install) section to add a new project using existing GitHub sources option and enter `clusteorne/clusterone-tutorials`. 

Let's create a job. Make sure to replace `username` with your username.

```shell
just create job distributed \
    --project username/clusterone-tutorials \
    --name horovod-job \
    --docker-image horovod-0.15.0-cpu-py36-tf1.11.0 \
    --ps-docker-image horovod-0.15.0-cpu-py36-tf1.11.0 \
    --worker-type t2.small \
    --ps-type t2.small \
    --worker-replicas 2 \
    --ps-replicas 1 \
    --time-limit 1h \
    --command "mpirun --allow-run-as-root -np 3 --hostfile /kube-openmpi/generated/hostfile -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib python openmpi/horovod/main.py" \
    --setup-command "pip install clusterone"
```

Now all that's left to do is starting the job:

```shell
just start job clusterone-tutorials/horovod-job
```

That's it! You can monitor its progress on the command line using `just get events`. More elaborate monitoring is available on the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## More info
Horovod's GitHub [repo](https://github.com/uber/horovod/blob/master/docs/running.md) has good explanation of the -mca params. They also have good set of [examples](https://github.com/uber/horovod/tree/master/examples).

For tutorials on other distributed machine learning training, see [here](https://clusterone.com/tutorials).

If you have any further questions, don't hesitate to ask us on [Slack](https://bit.ly/2OPc6JH)!

## License

[MIT](LICENSE) © ClusterOne Inc.
