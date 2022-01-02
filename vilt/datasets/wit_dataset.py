import torch
import os
import re
import json
import h5py
import random
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
        assert split in ["TRAIN", "VAL", "TEST"]
        self.split = split
        self.data_dir = data_dir
        self.transforms = keys_to_wit_transforms(transform_keys)
        self.max_text_len = max_text_len

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(self.data_dir, self.split + '_IMAGES_' + 'wit_100_min_word_freq' + '.hdf5'), 'r')

        with open(os.path.join(self.data_dir, split + '_IMAGEIDS_wit_100_min_word_freq.json'), 'r') as j:
            self.image_ids = json.load(j)

        with open(os.path.join(self.data_dir, split + '_RAWSTRDESCS_wit_100_min_word_freq.json'), 'r') as j:
            self.str_descriptions = json.load(j)

        with open(os.path.join(self.data_dir, split + '_RAWSTRCAPS_wit_100_min_word_freq.json'), 'r') as j:
            self.str_captions = json.load(j)

        self.dataset_size = len(self.str_captions)

        self.draw_false_image = draw_false_image

    def open_hdf5(self):
        self.h = h5py.File('img.hdf5', 'r')

    def __getitem__(self, index):
        if not hasattr(self, 'h'):
            self.open_hdf5()
        img = torch.FloatTensor(self.h['images'][index] / 255.)

        if self.transforms is not None:
            img = self.transforms[0](img)
        img_id = torch.LongTensor([self.image_ids[index]])
        str_description = self.str_descriptions[index]
        str_caption = self.str_captions[index]
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
        image = torch.FloatTensor(self.h['images'][random_index] / 255.)
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

