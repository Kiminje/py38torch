from glob import glob
import pandas as pd
import os
from pathlib import Path
import numpy as np
print(pd.__version__)
data_pat = Path('./kaggle_bee_vs_wasp')
data_labels = pd.read_csv("/home/inje/py38torch/Machine_Learning/kaggle_bee_vs_wasp/labels.csv")
# data_labels = data_labels.set_index('id')
col = data_labels.id
data_labels = data_labels.reindex(np.random.permutation(data_labels.index))
# print(data_labels[1])
# print(col[0:5])
print(data_labels.is_bee[0: 10])
train = data_labels[(data_labels.is_validation == 0) & (data_labels.is_final_validation == 0)]
val = data_labels[data_labels.is_validation == 1]
test = data_labels[data_labels.is_final_validation == 1]
print(data_labels.head(5))
import Augmentor
p = Augmentor.DataFramePipeline(data_labels.query('(is_validation == 0) & (is_final_validation == 0)'),
                                image_col='path',
                                category_col='label')

p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.crop_random(probability=1, percentage_area=0.5)
print(p.status())