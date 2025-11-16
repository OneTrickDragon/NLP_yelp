import numpy as np
import pandas as pd
import re
from argparse import Namespace
import collections  
import os

args = Namespace(
    raw_train_dataset ='yelp/raw_train.csv',
    raw_test_dataset='yelp/raw_test.csv',
    train_proportion=0.7,
    val_proportion=0.3,
    output_csv='yelp/reviews_full.csv',
    seed = 9248
)

train_reviews = pd.read_csv(args.raw_train_dataset, header=None, names=['rating', 'review'])
train_reviews = train_reviews.dropna(subset=["review"])
test_reviews = pd.read_csv(args.raw_test_dataset, header=None, names=['rating', 'review'])
test_reviews = test_reviews.dropna(subset=["review"])

by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())

final_list = []
np.random.seed(args.seed)

for _, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_proportion * n)
    n_val = int(args.val_proportion * n)
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train + n_val]:
        item['split'] = 'val'
    final_list.extend(item_list)

for _, row in test_reviews.iterrows():
    row_dict = row.to_dict()
    row_dict['split'] = 'test'
    final_list.append(row_dict)

final_reviews = pd.DataFrame(final_list)
final_reviews = final_reviews.dropna(subset=["review"])
final_reviews.to_csv(args.output_csv, index=False)
