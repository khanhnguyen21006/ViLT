import torch
import os
import re
import json
import pyarrow as pa
# import h5py
import random
import io
from PIL import Image
import base64
import numpy as np
import cv2
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vilt.transforms import keys_to_wit_transforms

SPACE_NORMALIZER = re.compile(r"\s+")


class WitDataset(Dataset):
    def __init__(
            self,
            split: str,
            data_dir: str,
            transform_keys: list,
            max_text_len: int,
            draw_false_image: int = 0,
    ):
        assert split.lower() in ["train", "val", "test"]
        self.split = split.lower()
        self.data_dir = data_dir
        self.transforms = keys_to_wit_transforms(transform_keys)
        self.max_text_len = max_text_len

        # Open hdf5 file where images are stored

        # with open(os.path.join(self.data_dir, self.split + '_IMAGEIDS_wit_100_min_word_freq.json'), 'r') as j:
        #     self.image_ids = json.load(j)
        #
        # with open(os.path.join(self.data_dir, self.split + '_RAWSTRDESCS_wit_100_min_word_freq.json'), 'r') as j:
        #     self.str_descriptions = json.load(j)
        #
        # with open(os.path.join(self.data_dir, self.split + '_RAWSTRCAPS_wit_100_min_word_freq.json'), 'r') as j:
        #     self.str_captions = json.load(j)

        self.draw_false_image = draw_false_image

        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}wit_{self.split}.arrow", "r")).read_all()

        self.imgs = self.table["image"].to_pandas().tolist()
        self.str_captions = self.table["caption"].to_pandas().tolist()
        self.str_descriptions = self.table["context"].to_pandas().tolist()
        self.image_ids = self.table["image_id"].to_pandas().tolist()

        self.dataset_size = len(self.str_captions)

    # def open_hdf5(self):
    #     self.h = h5py.File(os.path.join(self.data_dir, self.split + '_IMAGES_' + 'wit_100_min_word_freq' + '.hdf5'), 'r')
    #     self.imgs = self.h['images']

    def __getitem__(self, index):
        # if not hasattr(self, 'h'):
        #     self.open_hdf5()
        try:
            base64_decoded = base64.b64decode(self.imgs[index])
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
            print(f'{e} at image {index}')
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
        except Exception as e:
            print(f"An exception occurred {e} at image {index}")

        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 256, 256)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)

        if self.transforms is not None:
            img = self.transforms[0](img)
        img_id = torch.LongTensor([self.image_ids[index]])
        str_description = self.str_descriptions[index][0]
        str_caption = self.str_captions[index][0]
        context = self.get_text(str_description)
        caption = self.get_text(str_caption)

        ret = {
            "image": img,
            "image_id": img_id,
            "context": context["text"],
            "caption": caption["text"],
        }

        for fi in range(self.draw_false_image):
            false_img = self.get_false_image(fi)
            ret.update(false_img)

        return ret

    def __len__(self):
        return self.dataset_size

    def get_false_image(self, rep):
        random_index = random.randint(0, self.dataset_size - 1)
        try:
            base64_decoded = base64.b64decode(self.imgs[random_index])
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
            print(f'{e} at image {random_index}')
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
        except Exception as e:
            print(f"An exception occurred {e} at image {random_index}")

        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 256, 256)
        assert np.max(img) <= 255
        image = torch.FloatTensor(self.imgs[random_index] / 255.)
        if self.transforms is not None:
            image = self.transforms[0](image)
        return {f"false_image_{rep}": image}

    def get_text(self, sentence):
        encoding = self.to_token_ids(sentence)
        return {
            "text": (sentence, encoding),  # (str, dict)
        }

    def to_token_ids(self, sentence):
        bpe_tokens = self.tokenizer.bpe.encode(sentence)
        bpe_tokens = SPACE_NORMALIZER.sub(" ", bpe_tokens)
        words = bpe_tokens.strip().split()

        words = words[:self.max_text_len - 2]
        words = ['<s>'] + words + ['</s>']

        token_ids = []
        for word in words:
            idx = self.tokenizer.task.source_dictionary.indices[word]
            token_ids.append(idx)

        token_ids = [self.tokenizer.task.source_dictionary.indices[word] for word in words]

        return torch.LongTensor(token_ids)

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        for img_key in img_keys:
            dict_batch[img_key] = torch.stack(dict_batch[img_key])

        txt_keys = ["context", "caption"]
        if len(txt_keys) != 0:
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )
                padded_encodings = pad_sequence(encodings, batch_first=True, padding_value=1.0).long()
                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = padded_encodings
                if txt_key == "caption":
                    dict_batch[f"{txt_key}_masks"] = padded_encodings != 1
        return dict_batch

