import torch
import os
import re
import json
import h5py
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vilt.transforms import keys_to_wit_transforms
import spacy

SPACE_NORMALIZER = re.compile(r"\s+")
nlp = spacy.load("en_core_web_lg")


class WitDataset(Dataset):
    def __init__(
            self,
            split: str,
            data_dir: str,
            transform_keys: list,
            max_text_len: int,
            nmlm: int = 0,
            draw_false_image: int = 0,
    ):
        assert split in ["TRAIN", "VAL", "TEST"]
        self.split = split
        self.data_dir = data_dir
        self.transforms = keys_to_wit_transforms(transform_keys)
        self.max_text_len = max_text_len

        with open(os.path.join(self.data_dir, self.split + '_IMAGEIDS_wit_100_min_word_freq.json'), 'r') as j:
            self.image_ids = json.load(j)

        with open(os.path.join(self.data_dir, self.split + '_IMAGEURLS_wit_100_min_word_freq.json'), 'r') as j:
            self.image_urls = json.load(j)

        with open(os.path.join(self.data_dir, self.split + '_STRDESCS_wit_100_min_word_freq.json'), 'r') as j:
            self.str_descriptions = json.load(j)

        with open(os.path.join(self.data_dir, self.split + '_STRCAPS_wit_100_min_word_freq.json'), 'r') as j:
            self.str_captions = json.load(j)

        self.dataset_size = len(self.str_captions)

        self.nmlm = nmlm
        self.draw_false_image = draw_false_image

    def open_hdf5(self):
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(self.data_dir, self.split + '_IMAGES_' + 'wit_100_min_word_freq' + '.hdf5'), 'r')
        self.imgs = self.h['images']

    def __getitem__(self, index):
        if not hasattr(self, 'h'):
            self.open_hdf5()
        img = torch.FloatTensor(self.imgs[index] / 255.)

        if self.transforms is not None:
            img = self.transforms[0](img)
        img_id = torch.LongTensor([self.image_ids[index]])
        str_description = self.str_descriptions[index]
        str_caption = self.str_captions[index]
        str_description = ' '.join(str_description).strip()
        str_caption = ' '.join(str_caption).strip()
        context = self.get_text(str_description)
        caption = self.get_text(str_caption, self.nmlm)
        image_url = self.image_urls[index]
        ret = {
            "image": img,
            "image_id": img_id,
            "image_url": image_url,
            "context": context["text"],
            "caption": caption["text"],
        }
        if self.nmlm:
            ret.update({
                "caption_nmlm": caption["masked_text"],
            })

        for fi in range(self.draw_false_image):
            false_img = self.get_false_image(fi)
            ret.update(false_img)

        return ret

    def __len__(self):
        return self.dataset_size

    def get_false_image(self, rep):
        random_index = random.randint(0, self.dataset_size - 1)
        image = torch.FloatTensor(self.imgs[random_index] / 255.)
        if self.transforms is not None:
            image = self.transforms[0](image)
        return {f"false_image_{rep}": image}

    def get_text(self, sentence, nmlm=0):
        ret = {}
        if nmlm:
            encoding, masked_encoding = self.to_masked_ner_token_ids(sentence)
            ret.update({
                "text": (sentence, encoding),
                "masked_text": (sentence, masked_encoding),
            })
        else:
            encoding = self.to_token_ids(sentence)
            ret.update({
                "text": (sentence, encoding),
            })
        return ret

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

        img_keys = [k for k in list(dict_batch.keys()) if "_image_" in k or k == "image"]
        for img_key in img_keys:
            dict_batch[img_key] = torch.stack(dict_batch[img_key])

        txt_keys = [k for k in list(dict_batch.keys()) if "context" in k or "caption" in k]
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

    def to_masked_ner_token_ids(self, sentence):
        bpe_tokens = []
        bpe_ner_masks = []

        raw_tokens = self.tokenizer.bpe.bpe.re.findall(self.tokenizer.bpe.bpe.pat, sentence)
        text_doc = nlp(sentence)
        ner_masks = self.get_entity_mask(raw_tokens, text_doc)
        assert len(ner_masks) == len(raw_tokens)

        for raw_token, ner_mask in zip(raw_tokens, ner_masks):
            token = ''.join(self.tokenizer.bpe.bpe.byte_encoder[b] for b in raw_token.encode('utf-8'))
            # e.g. token == "ĠTomas"

            token_ids = [self.tokenizer.bpe.bpe.encoder[bpe_token] for bpe_token in self.tokenizer.bpe.bpe.bpe(token).split(' ')]
            # e.g. token_ids == [6669, 959]

            # bpe_raw_tokens.extend(self.tokenizer.bpe.bpe(token).split(' '))
            bpe_tokens.extend(token_ids)

            bpe_ner_masks.extend([1] * len(token_ids) if ner_mask else [0] * len(token_ids))

        assert bpe_tokens == self.tokenizer.bpe.bpe.encode(sentence)

        bpe_tokens = SPACE_NORMALIZER.sub(" ", ' '.join(map(str, bpe_tokens)))
        words = bpe_tokens.strip().split()
        assert len(words) == len(bpe_ner_masks)

        words = words[:self.max_text_len - 2]
        words = ['<s>'] + words + ['</s>']
        ner_mask = [0] + bpe_ner_masks + [0]

        token_ids = []
        masked_ner_token_ids = []
        for i, word in enumerate(words):
            token_ids.append(self.tokenizer.task.source_dictionary.indices[word])
            masked_ner_token_ids.append(self.tokenizer.task.source_dictionary.indices[word] if ner_mask[i] == 0 else 50264)

        return torch.LongTensor(token_ids), torch.LongTensor(masked_ner_token_ids)

    def get_entity_mask(self, tokens, doc):
        # We first compute the start and end points for each token.
        # End points are exclusive.
        # e.g. tokens = [' Tomas', ' Maier', ',', ' autumn', '/', 'winter', ' 2014', ',', '\n', ' in', 'Milan', '.']
        starts = []
        ends = []
        current = 0
        for token in tokens:
            starts.append(current)
            current += len(token)
            ends.append(current)

        copy_masks = [0] * len(tokens)

        if doc is None:
            return copy_masks

        # Next we get the character positions of named entities
        for ent in doc.ents:
            if random.random() < 0.8:
                # A token is part of an entity if it lies strictly inside it
                for i, (start, end, token) in enumerate(zip(starts, ends, tokens)):
                    entity_start = ent.start_char
                    if token[0] == ' ':
                        entity_start -= 1
                    entity_end = ent.end_char

                    if start >= entity_start and end <= entity_end and ent.label_ in ['PERSON', 'ORG', 'GPE']:
                        copy_masks[i] = 1
        return copy_masks
