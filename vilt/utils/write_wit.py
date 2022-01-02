import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
import h5py


def make_arrow(root, dataset_root):
    # Open hdf5 file where images are stored
    split = 'TEST'

    h = h5py.File(os.path.join(root, split + '_IMAGES_' + 'wit_100_min_word_freq' + '.hdf5'), 'r')
    imgs = h['images']

    with open(os.path.join(root, split + '_IMAGEIDS_wit_100_min_word_freq.json'), 'r') as j:
        image_ids = json.load(j)

    with open(os.path.join(root, split + '_RAWSTRDESCS_wit_100_min_word_freq.json'), 'r') as j:
        str_descriptions = json.load(j)

    with open(os.path.join(root, split + '_RAWSTRCAPS_wit_100_min_word_freq.json'), 'r') as j:
        str_captions = json.load(j)

    dataset_size = len(str_captions)

    assert len(imgs) == len(image_ids) == len(str_descriptions) == len(str_captions) == dataset_size

    shuffled_ids = [i for i in range(dataset_size)]
    random.shuffle(shuffled_ids)

    batches = [[pa.Tensor(), image_ids[s_id], str_descriptions[s_id], str_captions[s_id]] for s_id in tqdm(shuffled_ids)]

    dataframe = pd.DataFrame(
        batches, columns=["image", "image_id", "context", "caption"],
    )

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
            f"{dataset_root}/wit_{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


