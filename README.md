
**Installing**

I recommend installing [Anaconda](https://www.continuum.io/downloads) to manage dependencies in isolated *environments*. Once you have downloaded the latest version of Anaconda for your OS, you can use the provided `environment.yml` file to automatically install the latest compatible packages for running the scripts in this repo.

Simply run:

```
conda env create -f environment.yml

(downloads packages)

source activate tf101gpu-py35
```

There are three different files for training:

* __train_ph_iou__ - train by manually feeding batches of images through tensorflow placeholders mechanism, uses intersection-over-union loss approximation from http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf - only supports binary classification
* __train_ph_xent__ - train by manually feeding batches of images through tensorflow placeholders mechanism, uses classic cross-entropy loss and supports arbitrary number of classes through `--out` argument. 
* __train_q__ - train by tensorflow queue prefetching mechanism, used for normal RGB images. Supports both loss types. 

Usage:

```
usage: train_*.py [-h] [--train_dir TRAIN_DIR] [--restore RESTORE]
                       [--gpu GPU] [--out OUT] [--bs BS] [--lr LR]
                       [--model MODEL] [--loss LOSS] [--fmt FMT] [--plot PLOT]
                       [--root_path ROOT_PATH] [--debug] [--email]
                       save
```

where:

* *MODEL* defines the size of the filter used in the VGG16 network. It can be xs, s, or '' for VGG16 with the smallest filter (filter size 16X32), VGG16 with filter size 32X64, or VGG16 with a 64X128 filter. Default is ''.
* *LOSS* defines the loss function used to optimize the network. It can be iou for Intersection-over-Union loss or xent for cross-entropy loss. Default is iou.
* *FMT* defines the format of the input images. It can be 'lab' for cielab images, or 'rgb'. Default is lab.
* *PLOT* is a boolean to show predictions every 100 steps. It can be either True or False. Default is False. 
* *save* is the subdir to log model and events.

The annotated dataset used with this code can be found at: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP/NTUOK9

To use the `--email` functionality, create a `.env` file in the root of
this repo with the following format:

```
EMAIL:<user>@gmail.com
PASSWORD:<password>
```

Note that only gmail is supported for now. If you are a UOGuelph MLRG member ask me for permission to use the noreply.jobs.gpu@gmail.com account so that you don't have to insecure your personal gmail. 
