import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import jpeg4py
import pandas as pd
import numpy as np
import torch


cv2.setNumThreads(1)
DATA_ROOT = Path(__file__).parent.parent / 'data'
TRAIN_ROOT = DATA_ROOT / 'train_images'
TEST_ROOT = DATA_ROOT / 'test_images'

UNICODE_MAP = {codepoint: char for codepoint, char in
               pd.read_csv(DATA_ROOT / 'unicode_translation.csv').values}


SEG_FP = 'seg_fp'  # false positive from segmentation


def load_train_df(path=DATA_ROOT / 'train.csv'):
    df = pd.read_csv(path)
    df['labels'].fillna(value='', inplace=True)
    return df


def load_train_valid_df(fold: int, n_folds: int):
    df = load_train_df()
    df['book_id'] = df['image_id'].apply(get_book_id)
    book_ids = np.array(sorted(set(df['book_id'].values)))
    with_counts = list(zip(
        book_ids,
        df.groupby('book_id')['image_id'].agg('count').loc[book_ids].values))
    with_counts.sort(key=lambda x: x[1])
    valid_book_ids = [book_id for i, (book_id, _) in enumerate(with_counts)
                      if i % n_folds == fold]
    train_book_ids = [book_id for book_id in book_ids
                      if book_id not in valid_book_ids]
    return tuple(df[df['book_id'].isin(ids)].copy()
                 for ids in [train_book_ids, valid_book_ids])


def get_image_path(item, root: Path = None) -> Path:
    if root is None:
        root = TEST_ROOT if item.image_id.startswith('test_') else TRAIN_ROOT
    path = root / f'{item.image_id}.jpg'
    assert path.exists(), path
    return path


def read_image(path: Path) -> np.ndarray:
    if path.parent.name == 'train_images':
        np_path = get_image_np_path(path)
        if np_path.exists():
            return np.load(np_path)
    return jpeg4py.JPEG(str(path)).decode()


def get_image_np_path(path):
    return path.parent / f'{path.stem}.npy'


def get_target_boxes_labels(item):
    if item.labels:
        labels = np.array(item.labels.split(' ')).reshape(-1, 5)
    else:
        labels = np.zeros((0, 5))
    boxes = labels[:, 1:].astype(np.float)
    labels = labels[:, 0]
    return boxes, labels


def get_encoded_classes() -> Dict[str, int]:
    classes = {SEG_FP}
    df_train = load_train_df()
    for s in df_train['labels'].values:
        x = s.split()
        classes.update(x[i] for i in range(0, len(x), 5))
    return {cls: i for i, cls in enumerate(sorted(classes))}


def get_book_id(image_id):
    book_id = re.split(r'[_-]', image_id)[0]
    m = re.search(r'^[a-z]+', book_id)
    if m:
        return m.group()
    else:
        return book_id


def to_coco(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert from pytorch detection format to COCO format.
    """
    boxes = boxes.clone()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


def from_coco(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert from CODO to pytorch detection format.
    """
    boxes = boxes.clone()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def scale_boxes(
        boxes: torch.Tensor, w_scale: float, h_scale: float) -> torch.Tensor:
    return torch.stack([
        boxes[:, 0] * w_scale,
        boxes[:, 1] * h_scale,
        boxes[:, 2] * w_scale,
        boxes[:, 3] * h_scale,
        ]).t()


def submission_item(image_id, prediction):
    return {
        'image_id': image_id,
        'labels': ' '.join(
            ' '.join([p['cls']] +
                     [str(int(round(v))) for v in p['center']])
            for p in prediction),
    }


def get_sequences(
        boxes: List[Tuple[float, float, float, float]],
        ) -> List[List[int]]:
    """ Return a list of sequences from bounding boxes.
    """
    # TODO expand tall boxes (although this is quite rarely needed)
    boxes = np.array(boxes)
    next_indices = {}
    for i, box in enumerate(boxes):
        x0, y0, w, h = box
        x1, _ = x0 + w, y0 + h
        bx0 = boxes[:, 0]
        bx1 = boxes[:, 0] + boxes[:, 2]
        by0 = boxes[:, 1]
        by1 = boxes[:, 1] + boxes[:, 3]
        w_intersecting = (
            ((bx0 >= x0) & (bx0 <= x1)) |
            ((bx1 >= x0) & (bx1 <= x1)) |
            ((x0 >= bx0) & (x0 <= bx1)) |
            ((x1 >= bx0) & (x1 <= bx1))
        )
        higher = w_intersecting & (by0 < y0)
        higher_indices, = higher.nonzero()
        if higher_indices.shape[0] > 0:
            closest = higher_indices[np.argmax(by1[higher_indices])]
            next_indices[closest] = i
    next_indices_values = set(next_indices.values())
    starts = {i for i in range(len(boxes)) if i not in next_indices_values}
    sequences = []
    for i in starts:
        seq = [i]
        next_idx = next_indices.get(i)
        while next_idx is not None:
            seq.append(next_idx)
            next_idx = next_indices.get(next_idx)
        sequences.append(seq)
    return sequences
