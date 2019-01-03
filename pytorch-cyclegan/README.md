# Distributed TensorFlow with Estimators

<p align="center">
<img src="../co_logo.png" alt="Clusterone" width="200">
<br>

This is a tutorial on how to train [CycleGAN](https://arxiv.org/abs/1703.10593) using distributed [PyTorch](https://pytorch.org/).

The tutorial itself is published on our [blog](https://clusterone.com/tutorials). You can read it [here](https://clusterone.com/tutorials/pytorch-cyclegan).

Follow the instructions below to run the tutorial code locally and on Clusterone. 

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [More Info](#more-info)
- [License](#license)

## Install

To run the code locally, you need:

- [Python](https://python.org/) 3.6
- [Git](https://git-scm.com/)
- PyTorch & Torchvision nightly. Install it using instructions on [https://pytorch.org/](https://pytorch.org/)
- Pillow. Install it like this: `pip install pillow`
- The Clusterone Python library. Install it with `pip install clusterone`

To run this project on Clusterone, you need:
- Clusterone account. Create a free account on [https://clusterone.com/](https://clusterone.com/).

That's all you need! Add a project by linking this GitHub repo (`clusterone/clusterone-tutorials`) as shown [here](https://docs.clusterone.com/documentation/projects-on-clusterone/github-projects#create-a-project-using-existing-github-repository).

### Data
You need some data to run this training. You can find some examples from the original CycleGAN authors [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

It can be any two sets of data that are similar (horses and zebras). You can even create your own.
Make sure the structure of your data directory is as follows:
```
/path/to/data/
├-- trainA
|   ├-- imageA-00000.jpeg
|   ├-- imageA-00001.jpeg
|   └-- ...
├-- trainB
|   ├-- imageB-00000.jpeg
|   └-- ...
├-- testA
|   ├-- imageA-10000.jpeg
|   ├-- imageA-10001.jpeg
|   └-- ...
└-- testB
    ├-- imageB-10000.jpeg
    └-- ...

```

Locally, you can place this dataset anywhere, just remember the path.
On Clusterone, you'll have to create a dataset. You can use AWS S3, GitHub, or GitLab. Follow the instructions [here](https://docs.clusterone.com/documentation/data-on-clusterone).

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

Make sure you have all requirements installed that are listed above. Also, make sure you have a dataset ready (I will assume the data is in directory `/data/`.)

Here is command you can run to start training:
```shell
python pytorch-cyclegan/main.py --data_dir /data/ --batch_size 1 --print_steps 1 --save_steps 2
```
If you have a compatible GPU on your computer, I recommend increasing the print_steps & save_steps. Increasing the batch_size is not recommended, unless you have more than 12GB of memory.

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
| 0 | username/clusterone-tutorials | 2019-01-02T01:23:45 |             |
```
where `username` should be your Clusterone account name.

Let's create a job. Make sure to replace `username` with your username and `dataset-name` with your dataset name.
```shell
just create job distributed \
  --project username/clusterone-tutorials \
  --datasets username/dataset-name \
  --name distributed-cyclegan-job \
  --worker-replicas 2 \
  --worker-type p2.xlarge \
  --docker-image pytorch-latest-gpu-py36-cuda9.0 \
  --ps-replicas 1 \
  --ps-type p2.xlarge \
  --ps-docker-image pytorch-latest-gpu-py36-cuda9.0 \
  --time-limit 2h \
  --command "python pytorch-cyclegan/main.py --data_dir /data/username/dataset-name/ --batch_size 1 --print_steps 2 --save_steps 5" \
  --setup-command "pip install clusterone"
```

This creates a job with 2 worker nodes and 1 parameter server. See our [documentation](https://docs.clusterone.com/cli-reference-documentation/just-create-job) for more information on different job parameters.

Now the final step is to start the job:

```shell
just start job clusterone-tutorials/distributed-cyclegan-job
```

That's it! You can monitor its progress on the command line using `just get events`. More elaborate monitoring is available on the [Matrix](https://clusterone.com/matrix), Clusterone's graphical web interface.

## More Info

For further information on this example, take a look at the [tutorial](https://clusterone.com/tutorials/pytorch-cyclegan) based on this repository on the Clusterone Blog.

The code is adapted from Jun-Yan Zhu's [GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), who is the author of the CycleGAN paper.

If you have any further questions, don't hesitate to reach out on [Slack](https://bit.ly/2OPc6JH)!

## License
For works from the original CycleGAN repo, please refer to LICENSE_CYCLEGAN.

[MIT](LICENSE) © Clusterone Inc.
