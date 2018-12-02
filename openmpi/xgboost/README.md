# XGBoost Tutorial

<p align="center">
<img src="../../co_logo.png" alt="Clusterone" width="200">
</p>

This tutorial presents an example to learn how to use distributed [XGBoost](https://xgboost.readthedocs.io).

This repository contains the code and data files required to run the tutorial model. For the tutorial itself, please [see here](https://clusterone.com/tutorial/openmpi-introduction).

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
- The XGBoost Python library. Install it using `pip install xgboost`
- The Clusterone Python library. Install it with `pip install clusterone`

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

Then cd into the xgboost folder with `cd clusterone-tutorials/openmpi/xgboost`.

Single instance mode is very simple. You just execute main.py:
```shell
python main.py
```

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
where `username` should be your Clusterone account name. If you don't have this project, then add a new project using existing github sources option and enter `clusteorne/clusterone-tutorials`. 

Let's create a job. Make sure to replace `username` with your username.

```shell
just create job distributed \
    --project username/clusterone-tutorials \
    --name xgboost-job \
    --docker-image xgboost-0.81-cpu-py36-openmpi3.1.3 \
    --ps-docker-image xgboost-0.81-cpu-py36-openmpi3.1.3 \
    --worker-type t2.small \
    --ps-type t2.small \
    --worker-replicas 2 \
    --ps-replicas 1 \
    --time-limit 1h \
    --command "/dmlc-core/tracker/dmlc-submit --cluster mpi --num-workers 3 python openmpi/xgboost/main.py" \
    --setup-command "pip install clusterone"
```

Now all that's left to do is starting the job:

```shell
just start job clusterone-tutorials/xgboost-job
```

That's it! You can monitor its progress on the command line using `just get events`. More elaborate monitoring is available on the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

#### Benchmark
If you're interested in running benchmarks, replace the Python command with the following (on Clusterone platform):
```shell
python openmpi/xgboost/main.py --benchmark --data_dir /public/xgboost-benchmark-dataset/
```

The official benchmark was done using AWS m5.large instances.

## More info
Read our [OpenMPI tutorial](https://clusterone.com/tutorials/openmpi-introduction).

For XGBoost's tutorial on distributed training, see [here](https://xgboost.readthedocs.io/en/latest/tutorials/aws_yarn.html).

For tutorials on other distributed machine learning training, see [here](https://clusterone.com/tutorials).

If you have any further questions, don't hesitate to ask us on [Slack](https://bit.ly/2OPc6JH)!

## License

[MIT](LICENSE) © ClusterOne Inc.