# TIRG Composing Text and Image for Image Retrieval with Bonn Furniture Styles dataset

by Jeffery Braga, Doris Chia-ching Lin

This project is an attempt to use TIRG function for image retrieval with new furniture style dataset. The originally code was published with the paper:

**<a href="https://arxiv.org/abs/1812.07119">Composing Text and Image for Image Retrieval - An Empirical Odyssey
</a>**
<br>
Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays
<br>
CVPR 2019.
**<a href="https://github.com/google/tirg">github code source</a>**

The dataset was published with the paper below:

**<a href="https://arxiv.org/abs/1812.07119">Learning Style Compatibility for Furniture
</a>**
<br>
Divyansh Aggarwal, Elchin Valiyev, Fadime Sener, and Angela Yao
<br>
CVPR 2018.
**<a href="https://cvml.comp.nus.edu.sg/furniture/download.html">dataset download</a>**

## Implementation

In order implement TRIG to the furniture style dataset, we have created a `test_queries.txt` to map the index of source image and target image for image retrieval. Each image has captions could be cross-referenced with testing queries. Here are our dataset stats:

Train Model

- FurnitureStyle: 133346 images
- 56520 unique cations
- Modifiable images 40076

Test Model
- FurnitureStyle: 28812 images
- 28812 test queries

## Setup

- torchvision
- pytorch
- numpy
- tqdm
- tensorboardX
- Python 3.0.0 or above

## Hardware Requirement:

- Need have a `cuda` **<a href="https://developer.nvidia.com/cuda-gpus">compatible graphic card
  </a>**

## Running Models

- `main.py`: driver script to run training/testing
- `datasets.py`: Dataset classes for loading images & generate training retrieval queries
- `text_model.py`: LSTM model to extract text features
- `img_text_composition_models.py`: various image text compostion models (described in the paper)
- `torch_function.py`: contains soft triplet loss function and feature normalization function
- `test_retrieval.py`: functions to perform retrieval test and compute recall performance

### FurnitureStyle dataset

Download our generated test_queries.txt from [here](furniture-style/test_queries.txt).

Make sure the dataset include these files:

```
<dataset_path>/splits/*.txt
<dataset_path>/houzz/<category>/<style>/*.jpeg
<dataset_path>/test_queries.txt`
```

note that the file name `val_split` in `/splits/` from origin dataset should be renamed to `test_val_split`

Run training & testing:

```
python main.py --dataset=furnitureStyle --dataset_path=./furniture-style \
  --num_iters=160000 --model=concat --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=furnitureS_concat

python3 main.py --dataset=furnitureStyle --dataset_path=./furniture-style \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=furnitureS_tirg
```
