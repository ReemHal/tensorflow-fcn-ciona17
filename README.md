Usage:

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
