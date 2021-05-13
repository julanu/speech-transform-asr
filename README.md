# Speech Transformer: End-to-End ASR with Transformer

## Notes
In order to run this project, you should either use a dual-boot machine, with any *nix system of your choice, or you could use WSL2 on Windows(Ubuntu20.04 on Windows).
The algorithm makes use of the NVIDIA drivers for training and processing, thus you can not make those drivers available on a guest box(VM) without making it available only for the guest, meaning it won't be shared with the host machine. This is rarely possible so the best setup would be a Linux machine with dedicated GPUs for this architecture.
<br />
<br />

One chance to skip the dual boot part, though not feasible(at this moment April 2021), would be:<br/>
[CUDA on WSL :: CUDA Toolkit Documentation (nvidia.com)](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

[Getting started with CUDA on Ubuntu WSL2](https://ubuntu.com/blog/getting-started-with-cuda-on-ubuntu-on-wsl-2).

I would also suggest checking out this StackOverflow [answer](https://askubuntu.com/questions/1252964/please-help-configuring-nvidia-smi-ubuntu-20-04-on-wsl-2) if you have problems setting up your own machine.

### If you want to use WSL2:
-> Ensure that you install Build version 20145 or higher. You can check your build version number by running `winver` via the Windows Run command.


## Installation

Clone the repo of the projects from [here](https://github.com/kaituoxu/Speech-Transformer). But besides the installation steps provided by the projects maintainer, there are some additional steps that may have been missed.

```git
git clone https://github.com/kaituoxu/Speech-Transformer
```

The following additional dependencies should be installed, note that PyTorch was installed with the CPU version for CPU training, otherwise you can install as normal the PyTorch package:<br />
```python
certifi==2020.12.5
chardet==4.0.0
dataclasses==0.6
future==0.18.2
idna==2.10
jsonpatch==1.32
jsonpointer==2.1
kaldi-io==0.9.4
numpy
Pillow==8.1.2
pyzmq==22.0.3
requests==2.25.1
scipy
six==1.15.0
torch==1.7.0+cpu
torchaudio==0.7.0
torchfile==0.1.0
torchvision==0.8.1+cpu
tornado==6.1
typing-extensions==3.7.4.3
urllib3==1.26.4
visdom==0.1.8.9
websocket-client==0.58.0
```
You can achieve this with: ```pip3 install -r requirements.txt```<br />

After cloning the repo, copy the [KALDI](https://github.com/kaldi-asr/kaldi) repo inside the `tools` folder of you Speech-Transformer copy:

```shell
cd Speech-Transformer/tools
git clone https://github.com/kaldi-asr/kaldi
```

After changing your directory to `tools`, follow the installation instructions for KALDI as per the `INSTALL` files present there.
You can always check if you have all the dependencies by running:
```
Speech-Transformer/tools/kaldi/tools$ bash extras/check_dependencies.sh
```

You should run the following command, this being in the output of `extras/check_dependencies.sh` and (probably) if you don't have them installed this is what you will run:
```
user@Speech-Transformer/tools/kaldi/tools$ sudo apt-get install automake autoconf sox gfortran libtool subversion python2.7
# Additionally the mkl software package is installed too
user@Speech-Transformer/tools/kaldi/tools$ bash extras/install_mkl.sh
```
After you've installed all dependencies necessary, this is the output message you should see:
```
$ bash extras/check_dependencies.sh
extras/check_dependencies.sh: all OK.
```
The script `aishell_data_prep.sh`
is going to create all the training, testing and development directories; it's going to feed the training algorithm the characters one by one, each character having a token with which it is represented.

And it can be found under the following locations:
```
./Speech-Transformer/egs/aishell/data/local/test/transcripts.txt
./Speech-Transformer/egs/aishell/data/local/dev/transcripts.txt
./Speech-Transformer/egs/aishell/data/local/train/transcripts.txt
```
After the data is prepared and processed, it can be found under the following locations:
```
./Speech-Transformer/egs/aishell/dump/test/deltafalse/data.json
./Speech-Transformer/egs/aishell/dump/dev/deltafalse/data.json
./Speech-Transformer/egs/aishell/dump/train/deltafalse/data.json
```
The data provided to the training algorithm is as follows:
```python
train.py
-train-json dump/train/deltafalse/data.json   # Training data
-valid-json dump/dev/deltafalse/data.json     # Dev data for testing
-dict data/lang_1char/train_chars.txt         # Valid characters
[...]
```

There is this script, which configs how much memory/gpu some scripts should use, which you should take
into account:

```
$ cat egs/aishell/cmd.sh  

# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

export train_cmd="run.pl --mem 2G"
export cuda_cmd="run.pl --mem 2G --gpu 1"
export decode_cmd="run.pl --mem 4G"

# NPU setup
# export train_cmd="queue.pl -q all.q --mem 2G"
# export cuda_cmd="/home/work_nfs/common/tools/pyqueue_asr.pl --mem 2G --gpu 1"
# export decode_cmd="/home/work_nfs/common/tools/pyqueue_asr.pl --mem 4G --gpu 1"
#export cuda_cmd="queue.pl --mem 2G --gpu 1 --config conf/gpu.conf"
#export decode_cmd="queue.pl -q all.q --mem 4G"
```
Changing the values since I am using a system of 4vCPU and 16GB RAM:
```
export train_cmd="run.pl --mem 4G"
export cuda_cmd="run.pl --mem 4G --gpu 1"
export decode_cmd="run.pl --mem 6G"
```



For CPU training specifically, this is how you install the `PyTorch` package to the latest stable version(at this time):
```
pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```



A PyTorch implementation of Speech Transformer [1], an end-to-end automatic speech recognition with [Transformer](https://arxiv.org/abs/1706.03762) network, which directly converts acoustic features to character sequence using a single nueral network.


## Usage
Before you can start training the algorithm, you should modify the path to the dataset first:
```bash
$ cd egs/aishell
# Modify aishell data path to your path in the begining of run.sh, the you can execute it
$ bash run.sh
```

Download the dataset intended for this algorithm: [aishell](http://www.openslr.org/33/) dataset for free. <br />
You can also use one of the following datasets: [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)

You can change parameters by `$ bash run.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`.<br />
See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.

### Workflow of `egs/aishell/run.sh`:
- Stage 0: Data Preparation
- Stage 1: Feature Generation
- Stage 2: Dictionary and Json Data Preparation
- Stage 3: Network Training
- Stage 4: Decoding
### More detail you can find here: `egs/aishell/run.sh`
```bash
# Set PATH and PYTHONPATH
$ cd egs/aishell/; . ./path.sh
```
#### How to visualize loss?
If you want to visualize your loss, you can use [visdom](https://github.com/facebookresearch/visdom) to do that:
1. Open a new terminal in your remote server (recommend tmux) and run `$ visdom`.
2. Open a new terminal and run `$ bash run.sh --visdom 1 --visdom_id "<any-string>"` or `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`.
3. Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`.
4. In visdom website, chose `<any-string>` in `Environment` to see your loss.

<!-- ![loss](egs/aishell/figures/train-k0.2-bf15000-shuffle-ls0.1.png) -->
#### How to resume training?
In order to use this feature, a `checkpoint` folder should be setup for the algorithm, this can be done through
```bash
$ bash run.sh --continue_from <model-path>
```
#### How to solve out of memory?
This can happen while the algorithm is training, try to reduce the `batch_size` parameter in the `run.sh` script.<br />
`$ bash run.sh --batch_size <lower-value>`.



## Results
| Model | CER | Config |
| :---: | :-: | :----: |
| LSTMP | 9.85| 4x(1024-512). See [kaldi-ktnet1](https://github.com/kaituoxu/kaldi-ktnet1/blob/ktnet1/egs/aishell/s5/local/nnet1/run_4lstm.sh)|
| Listen, Attend and Spell | 13.2 | See [Listen-Attend-Spell](https://github.com/kaituoxu/Listen-Attend-Spell)'s egs/aishell/run.sh |
| SpeechTransformer | 12.8 | See egs/aishell/run.sh |

## Reference
- [1] Yuanyuan Zhao, Jie Li, Xiaorui Wang, and Yan Li. "The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition." ICASSP 2019.
