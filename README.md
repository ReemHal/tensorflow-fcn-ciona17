
**Installing**

I recommend installing [Anaconda](https://www.continuum.io/downloads) to manage dependencies in isolated *environments*. Once you have downloaded the latest version of Anaconda for your OS, you can use the provided `environment.yml` file to automatically install the latest compatible packages for running the scripts in this repo.

Simply run:

```
conda env create -f environment.yml

(downloads packages)

source activate tf101gpu-py35
```

There are three different files for training:

* __train_ph_iou.py__ - train by manually feeding batches of images through tensorflow placeholders mechanism, uses intersection-over-union loss approximation from http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf - only supports binary classification
* __train_ph_xent.py__ - train by manually feeding batches of images through tensorflow placeholders mechanism, uses classic cross-entropy loss and supports arbitrary number of classes through `--out` argument. 
* __train_q.py__ - train by tensorflow queue prefetching mechanism, used for normal RGB images. Supports both loss types. 

If using `train_q.py`, you must create tfrecords first:

training - should result in a 140.9MB \*.tfrecords file
```
python preprocessing/rgb_to_byte_tfrecords.py --in_path <path/to/>ciona17_farm1_training1/ --out_path <path/to/>tfrecord/ --rec ciona17-rgb-train
```
validation - should result in a 67.1MB \*.tfrecords file
```
python preprocessing/rgb_to_byte_tfrecords.py --in_path <path/to/>ciona17_farm1_validatio
n/ --out_path <path/to/>tfrecord/ --rec ciona17-rgb-valid
```


Then, usage of training scripts:
```
usage: train_*.py [-h] [--train_dir TRAIN_DIR] [--restore RESTORE]
                       [--gpu GPU] [--out OUT] [--bs BS] [--lr LR]
                       [--model MODEL] [--loss LOSS] [--fmt FMT] [--plot PLOT]
                       [--root_path ROOT_PATH] [--debug] [--email]
                       save
```

The annotated dataset used with this code can be found at: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP/NTUOK9

To use the `--email` functionality, create a `.env` file in the root of
this repo with the following format:

```
EMAIL:<user>@gmail.com
PASSWORD:<password>
```

Note that only gmail is supported for now. If you are a MLRG member ask me for permission to use the noreply.jobs.gpu@gmail.com account so that you don't have to insecure your personal gmail. 
