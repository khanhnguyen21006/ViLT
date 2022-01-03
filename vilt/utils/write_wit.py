import json
import os
import pandas as pd
import pyarrow as pa
import numpy as np
from PIL import Image
import base64
from tqdm import tqdm
import io
import cv2

field_as_context = 'tokenized_cad'
field_as_context_length = 'cad_length'
string_field_as_context = 'caption_attribution_description'
raw_desc_max_len = 200
desc_max_len = 512
cap_max_len = 100


def create_wit_input_files(split_df, split):
    image_descriptions = []
    raw_descriptions = []
    image_captions = []
    raw_captions = []
    image_ids = []
    image_urls = []

    for index, row in tqdm(split_df.iterrows()):

        captions = []
        descriptions = []
        temp_raw_descriptions = []
        temp_raw_captions = []

        if row[field_as_context_length] <= raw_desc_max_len:
            descriptions.append(row[field_as_context])
            row_raw_contex = row[string_field_as_context]
            if row_raw_contex.startswith('English: '):
                row_raw_contex = row_raw_contex.replace('English: ', '')
            temp_raw_descriptions.append(row_raw_contex.strip())

        if len(descriptions) == 0:
            continue

        if row.caption_length <= cap_max_len:
            captions.append(row.tokenized_caption)
            temp_raw_captions.append(row.caption_reference_description)

        if len(captions) == 0:
            continue

        image_descriptions.append(descriptions)
        raw_descriptions.append(temp_raw_descriptions)
        image_captions.append(captions)
        raw_captions.append(temp_raw_captions)
        image_ids.append(index)
        image_urls.append(row.image_url)

    # Sanity check
    assert len(image_descriptions) == len(image_captions) == len(image_ids) == len(image_urls) == len(
        raw_descriptions) == len(raw_captions)

    batches = []
    for imdescs, imrdescs, imcaps, imrcaps, imids, imurls, split in [
        (image_descriptions, raw_descriptions, image_captions, raw_captions, image_ids, image_urls, split)]:
        print("\nReading %s images, descriptions and captions, storing to file...\n" % split)

        for i, img_id in tqdm(enumerate(imids)):
            r_descriptions = imrdescs[i]
            r_captions = imrcaps[i]

            # Read images
            try:
                base64_decoded = base64.b64decode(split_df.loc[img_id, 'b64_bytes'])
                img = Image.open(io.BytesIO(base64_decoded))
                img = img.convert('RGB')
                img = np.array(img)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = np.array(Image.fromarray(img).resize((256, 256)))
                if len(img.shape) > 2 and img.shape[2] == 4:
                    # convert the image from RGBA2RGB for .png image
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except TypeError as e:
                print(f'{e} at image {img_id}')
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
            except Exception as e:
                print(f"An exception occurred {e} at image {img_id}")
                continue

            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255

            batches.append([split_df.loc[img_id, 'b64_bytes'], img_id, r_descriptions, r_captions])

    return batches


def make_arrow(root, dataset_root):
    # Open hdf5 file where images are stored
    print('======================================')
    print('Loading dataframe...')
    merged_df = pd.read_pickle(root + 'wit_en_merged_processed_mod.pkl')
    merged_df = merged_df.drop_duplicates(
        subset=['image_url', 'caption_attribution_description', 'caption_reference_description'])
    print(
        f'Total number of training samples (image_url, caption_attribution_description, caption_reference_description)',
        len(merged_df))

    with open(root + 'val_test_split.json', 'r') as vtsr:
        val_test = json.load(vtsr)
        print('Total validation + test data: ', len(val_test['images']))

    val = [img['imgid'] for img in val_test['images'] if img['split'] == 'val']
    test = [img['imgid'] for img in val_test['images'] if img['split'] == 'test']

    # val_df = merged_df.loc[val]
    # test_df = merged_df.loc[test]
    # train_df = merged_df.drop(test + val)

    print('Total training data: ', len(merged_df))
    print('-------------------------------------------------')
    print('|    Train    |    Validation     |    Test     |')
    print('-------------------------------------------------')
    print(f'|   {len(merged_df.drop(test + val))}    |        {len(merged_df.loc[val])}       |     {len(merged_df.loc[test])}    |')
    print('-------------------------------------------------')

    print('======================================')
    print('Creating data input files...')

    val_batches = create_wit_input_files(merged_df.loc[val], 'VAL')
    test_batches = create_wit_input_files(merged_df.loc[test], 'TEST')
    train_batches = create_wit_input_files(merged_df.drop(test + val).head(15000), 'TRAIN')

    val_dataframe = pd.DataFrame(
        val_batches, columns=["image", "image_id", "context", "caption"],
    )
    test_dataframe = pd.DataFrame(
        test_batches, columns=["image", "image_id", "context", "caption"],
    )
    train_dataframe = pd.DataFrame(
        train_batches, columns=["image", "image_id", "context", "caption"],
    )

    name_split = 'val'
    val_table = pa.Table.from_pandas(val_dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
            f"{dataset_root}/wit_{name_split}.arrow", "wb"
    ) as val_sink:
        with pa.RecordBatchFileWriter(val_sink, val_table.schema) as writer:
            writer.write_table(val_table)
    name_split = 'test'
    test_table = pa.Table.from_pandas(test_dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
            f"{dataset_root}/wit_{name_split}.arrow", "wb"
    ) as test_sink:
        with pa.RecordBatchFileWriter(test_sink, test_table.schema) as writer:
            writer.write_table(test_table)
    name_split = 'train'
    train_table = pa.Table.from_pandas(train_dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
            f"{dataset_root}/wit_{name_split}.arrow", "wb"
    ) as train_sink:
        with pa.RecordBatchFileWriter(train_sink, train_table.schema) as writer:
            writer.write_table(train_table)


