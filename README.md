# XAI DeepFake detection

## Environment
```
conda env create --file env.yml 
conda activate dfxai
```

## Checkpoints
Place in `checkpoints/` directory.

## About the models 
| Model | ff_attribution | swinv2_faceswap |
| --- | --- | --- |
| Task | multiclass | binary |
| Arch. | efficientnetv2_b0 | swinv2_tiny_window8_256 |
| Type | CNN | Vision Transformer |
| No. Params | 7.1M | 87.9M |
| No. Datasets | 1 | 5 |
| Input | (B, 3, 224, 224) | (B, 3, 256, 256) |
| Output | (B, 5) | (B, 1) |

### ff_attribution
Trained for multiclass classification on the FaceForensics++. Outputs a probability for each of the 5 classes, (0, 1, 2, 3, 4) corresponding to (real, neural textures, face2face, deepfakes, faceswap). The dataset includes both faceswap (deepfakes, faceswap) and face reenactment (neural textures, face2face) data.

#### Performance (FF++ test set)
| Metric | Value |
| --- | --- |
| MulticlassAccuracy | 0.9626 |
| MulticlassAUROC | 0.9970 |
| MulticlassF1Score | 0.9627 |
| MulticlassAveragePrecision | 0.9881 |

### swinv2_faceswap
Trained for binary classification (0 = real, 1 = fake) on faceswap data from DFDC, FF, FakeAV, ForgeryNet, and CelebDF. Outputs a single probability for the input being a deepfake. Probability 

#### Performance on faceswap data

| Metric \ Dataset | FF++ | FakeAVCeleb | CelebDF | ForgeryNet | DFDC | WildDeepFake | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUC | 0.9986 | 0.9995 | 0.9809 | 0.8627 | 0.8860 | 0.8556 | 0.9305 |
| F1 Score | 0.9917 | 0.9972 | 0.9586 | 0.6218 | 0.8860 | 0.8205 | 0.8793 |
| Balanced Accuracy | 0.9890 | 0.9921 | 0.8872 | 0.7538 | 0.7751 | 0.7355 | 0.8554 |
| FalsePositiveRate | 0.0121 | 0.0128 | 0.2083 | 0.0710 | 0.3065 | 0.3697 | 0.1634 |
| FalseNegativeRate | 0.0099 | 0.0030 | 0.0173 | 0.4214 | 0.1432 | 0.1593 | 0.1257 |


## Data Preprocessing
### FF++
1. Download dataset from [FaceForensics++](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)
2. Run the following script to preprocess the data:
```bash
conda activate dfxai
python3 src/data/preprocess_ff.py prepro -r RAW_DATA_PATH -tr PREPROCESSED_DATA_PATH -d cuda:0 -mdcsv RAW_DATA_PATH/dataset_info.csv -mdcsv faceforensics_frames.csv
```
Where `RAW_DATA_PATH` is the path to the downloaded FF++ dataset and `PREPROCESSED_DATA_PATH` is the path to save the preprocessed data. The script will create a new file `faceforensics_frames.csv` containing the paths to the preprocessed frames.

## Creating Database
1. Create a new LMDB database by running the following script:
```bash
conda activate dfxai
python3 src/data/lmdb_storage.py add-csv -csv ./faceforensics_frames.csv -h -pc relative_path -d ./ff.lmdb -ms 21474836480 -v -b PREPROCESSED_DATA_PATH
```
Where `faceforensics_frames.csv` is the file created in the previous step and `PREPROCESSED_DATA_PATH` is the path to the preprocessed data. The script will create a new LMDB database `ff.lmdb` containing the preprocessed frames. The `-ms` flag specifies the maximum size of the database in bytes, default is 20GB.