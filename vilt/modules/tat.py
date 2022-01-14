import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertConfig
from vilt.modules.linear import GehringLinear
from vilt.modules.attention import MultiHeadAttention
from vilt.modules.convolution import LightweightConv1dTBC, DynamicConv1dTBC
from vilt.modules import tat_heads, objectives, tat_utils, vilt_utils
from .resnet import resnet152


class TransformAndTell(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = resnet152()
        self.roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.large')
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False
        self.weigh_bert = config['weigh_bert']
        if self.weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(25))
            nn.init.uniform_(self.bert_weight)

        self.decoder = DynamicConvDecoder(config['embed_size'], config['embed_output_dim'], config['padding_idx'], config['init_size'], config['left_pad'],
                                          config['dropout'], config['decoder_conv_dim'], config['decoder_glu'], config['decoder_conv_type'],
                                          config['weight_softmax'], config['decoder_attention_heads'], config['weight_dropout'],
                                          config['relu_dropout'], config['input_dropout'], config['decoder_normalize_before'],
                                          config['attention_dropout'], config['decoder_ffn_embed_dim'], config['decoder_kernel_size_list'],
                                          config['decoder_layers'], config['final_norm'], config['vocab_size'], config['article_embed_size'])
        self.padding_idx = config['padding_idx']
        self.decoder.apply(self.init_weights)

        if config["loss_names"]["nmlm"] > 0:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["embed_output_dim"],
            )
            self.nmlm_score = tat_heads.CLMHead(bert_config)
            self.nmlm_score.apply(self.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.pooler = tat_heads.WitPooler(config["embed_output_dim"])
            self.pooler.apply(self.init_weights)
            self.itm_score = tat_heads.ITMHead(config["embed_output_dim"])
            self.itm_score.apply(self.init_weights)

        if config["loss_names"]["itm_wpa"] > 0:
            self.pooler = tat_heads.WitPooler(config["embed_output_dim"])
            self.pooler.apply(self.init_weights)
            self.itm_score = tat_heads.ITMHead(config["embed_output_dim"])
            self.wpa_embed = tat_heads.ITMWPAHead(config["embed_output_dim"])
            self.itm_score.apply(self.init_weights)
            self.wpa_embed.apply(self.init_weights)

        # ===================== Downstream ===================== #
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            print("Loading pretrained weights for fine-tuning...")
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

            for p in self.itm_score.parameters():
                p.requires_grad = False
            for p in self.pooler.parameters():
                p.requires_grad = False
            if config["loss_names"]["itm_wpa"] > 0:
                for p in self.wpa_embed.parameters():
                    p.requires_grad = False
            if config["loss_names"]["nmlm"] > 0:
                for p in self.nmlm_score.parameters():
                    p.requires_grad = False

        if config["loss_names"]["clm"] > 0:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["embed_output_dim"],
            )
            self.clm_score = tat_heads.CLMHead(bert_config)
            self.clm_score.apply(self.init_weights)

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            print("Loading pretrained weights for testing...")
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        # to be added in config
        self.sampling_topk = 1
        self.sampling_temp = 1

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            std = math.sqrt(1 / module.weight.shape[1])
            module.weight.data.normal_(mean=0, std=std)
            module.weight.data[self.padding_idx].fill_(0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "clm" in self.current_tasks:
            ret.update(objectives.compute_clm(self, batch))

        if "nmlm" in self.current_tasks:
            ret.update(objectives.compute_nmlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_tat_itm(self, batch))

        if "itm_wpa" in self.current_tasks:
            ret.update(objectives.compute_tat_itm_wpa(self, batch))

        return ret

    def infer(self, batch):

        do_nmlm = "_nmlm" if self.hparams.config["loss_names"]["nmlm"] > 0 else ""

        caption_ids = batch[f"caption{do_nmlm}_ids"]
        caption_masks = batch[f"caption_masks"]
        gt_caption_ids = batch["caption_ids"]
        target_ids = torch.zeros_like(gt_caption_ids)
        target_ids[:, :-1] = gt_caption_ids[:, 1:]
        target_ids = target_ids[:, :-1]

        # Embed the image
        image = batch["image"]
        X_image = self.resnet(image)
        # X_image.shape == [batch_size, 2048, 7, 7]

        X_image = X_image.permute(0, 2, 3, 1)
        # X_image.shape == [batch_size, 7, 7, 2048]

        # Flatten out the image
        B, H, W, C = X_image.shape
        P = H * W  # number of pixels
        X_image = X_image.view(B, P, C)
        # X_image.shape == [batch_size, 49, 2048]

        article_ids = batch["context_ids"]
        # article_ids.shape == [batch_size, seq_len]

        article_padding_mask = article_ids == self.padding_idx
        # article_padding_mask.shape == [batch_size, seq_len]

        B, S = article_ids.shape

        X_sections_hiddens = self.roberta.extract_features(
            article_ids, return_all_hiddens=True)

        if self.weigh_bert:
            X_article = torch.stack(X_sections_hiddens, dim=2)
            # X_article.shape == [batch_size, seq_len, 13, embed_size]

            weight = F.softmax(self.bert_weight, dim=0)
            weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
            # weight.shape == [1, 1, 13, 1]

            X_article = (X_article * weight).sum(dim=2)
            # X_article.shape == [batch_size, seq_len, embed_size]

        else:
            X_article = X_sections_hiddens[-1]
            # X_article.shape == [batch_size, seq_len, embed_size]

        # Create padding mask (1 corresponds to the padding index)
        image_padding_mask = X_image.new_zeros(B, P).bool()

        image_feats = X_image
        article_feats = X_article
        X_image = X_image.transpose(0, 1)
        X_article = X_article.transpose(0, 1)

        X = caption_ids
        X = self.decoder(X, X_image, image_padding_mask, X_article, article_padding_mask)

        X_feats, _ = X

        text_feats = X_feats[:, :-1, :]
        ret = {
            "text_feats": text_feats,
            "text_labels": target_ids,
            "text_ids": caption_ids,
            "text_masks": caption_masks,
            "image_feats": image_feats,
            "article_feats": article_feats,
            "image_padding_mask": image_padding_mask,
            "article_padding_mask": article_padding_mask,
        }
        if self.hparams.config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["itm_wpa"] > 0:
            batch_cls_out = []
            for b in range(caption_masks.shape[0]):
                batch_cls_out.append(X_feats[b][caption_masks[b], :][-1])
            cls_feats = self.pooler(torch.stack(batch_cls_out))
            ret.update({
                "cls_feats": cls_feats,
                "raw_cls_feats": torch.stack(batch_cls_out),
            })

        return ret

    def generate(self, batch, context):
        caption_ids = batch["caption_ids"]
        X_image = context["image_feats"].transpose(0, 1)
        X_article = context["article_feats"].transpose(0, 1)
        image_padding_mask = context["image_padding_mask"]
        article_padding_mask = context["article_padding_mask"]

        incremental_state = {}
        seed_input = caption_ids[:, 0:1]
        log_prob_list = []
        index_path_list = [seed_input]
        eos = 2
        active_idx = seed_input[:, -1] != eos
        full_active_idx = active_idx
        gen_len = 100
        B = caption_ids.shape[0]

        for i in range(gen_len):
            if i == 0:
                prev_target = seed_input
            else:
                prev_target = seed_input[:, -1:]

            self.decoder.filter_incremental_state(incremental_state, active_idx)

            decoder_out = self.decoder(prev_target, X_image[:, full_active_idx], image_padding_mask[full_active_idx], X_article[:, full_active_idx], article_padding_mask[full_active_idx], incremental_state=incremental_state)

            # We're only interested in the current final word
            decoder_out = decoder_out[0][:, -1:]

            # lprobs = self.decoder.get_normalized_probs(
            #     decoder_out, log_probs=True)
            # lprobs.shape == [batch_size, 1, vocab_size]
            lprobs = self.clm_score(decoder_out)

            lprobs = lprobs.squeeze(1)
            # lprobs.shape == [batch_size, vocab_size]

            topk_lprobs, topk_indices = lprobs.topk(self.sampling_topk)
            topk_lprobs = topk_lprobs.div_(self.sampling_temp)
            # topk_lprobs.shape == [batch_size, topk]

            # Take a random sample from those top k
            topk_probs = topk_lprobs.exp()
            sampled_index = torch.multinomial(topk_probs, num_samples=1)
            # sampled_index.shape == [batch_size, 1]

            selected_lprob = topk_lprobs.gather(
                dim=-1, index=sampled_index)
            # selected_prob.shape == [batch_size, 1]

            selected_index = topk_indices.gather(
                dim=-1, index=sampled_index)
            # selected_index.shape == [batch_size, 1]

            log_prob = selected_lprob.new_zeros(B, 1)
            log_prob[full_active_idx] = selected_lprob

            index_path = selected_index.new_full((B, 1), self.padding_idx)
            index_path[full_active_idx] = selected_index

            log_prob_list.append(log_prob)
            index_path_list.append(index_path)

            seed_input = torch.cat([seed_input, selected_index], dim=-1)

            is_eos = selected_index.squeeze(-1) == eos
            active_idx = ~is_eos

            full_active_idx[full_active_idx.nonzero()[~active_idx]] = 0

            seed_input = seed_input[active_idx]

            if active_idx.sum().item() == 0:
                break

        log_probs = torch.cat(log_prob_list, dim=-1)
        # log_probs.shape == [batch_size * beam_size, generate_len]

        token_ids = torch.cat(index_path_list, dim=-1)
        # token_ids.shape == [batch_size * beam_size, generate_len]

        return log_probs, token_ids

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        # ret = self(batch)
        ret = dict()

        # Embed the image
        image = batch["image"]
        X_image = self.resnet(image)
        X_image = X_image.permute(0, 2, 3, 1)
        B, H, W, C = X_image.shape
        P = H * W  # number of pixels
        X_image = X_image.view(B, P, C)
        article_ids = batch["context_ids"]
        article_padding_mask = article_ids == self.padding_idx
        B, S = article_ids.shape

        X_sections_hiddens = self.roberta.extract_features(article_ids, return_all_hiddens=True)

        X_article = torch.stack(X_sections_hiddens, dim=2)
        weight = F.softmax(self.bert_weight, dim=0)
        weight = weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        X_article = (X_article * weight).sum(dim=2)
        image_padding_mask = X_image.new_zeros(B, P).bool()
        image_feats = X_image
        article_feats = X_article

        context = {
            "image_feats": image_feats,
            "article_feats": article_feats,
            "image_padding_mask": image_padding_mask,
            "article_padding_mask": article_padding_mask,
        }
        ret.update({
            "context": context
        })

        if self.hparams.config["loss_names"]["clm"] > 0:
            ret.update(objectives.clm_test_step(self, batch, ret["context"]))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["clm"] > 0:
            objectives.clm_test_wrapup(outs, self.hparams.config["log_dir"])
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)


class DynamicConvDecoder(nn.Module):
    def __init__(self, embed_size, embed_output_dim, padding_idx, init_size,
                 left_pad, dropout, decoder_conv_dim, decoder_glu, decoder_conv_type,
                 weight_softmax, decoder_attention_heads, weight_dropout, relu_dropout,
                 input_dropout, decoder_normalize_before, attention_dropout,
                 decoder_ffn_embed_dim, decoder_kernel_size_list, decoder_layers=6,
                 final_norm=True, vocab_size=None, article_embed_size=1024):
        super().__init__()

        # word embedding
        self.word_embed = nn.Embedding(vocab_size, embed_size, padding_idx)
        self.word_embed_projection = nn.Linear(embed_size, embed_output_dim, bias=False)
        self.word_embed_seq = nn.Sequential(self.word_embed, self.word_embed_projection)

        # positional embedding
        self.pos_embed = tat_utils.SinusoidalPositionalEmbedding(init_size, embed_size, padding_idx, left_pad)

        self.dropout = dropout

        decoder_input_embed_dim = embed_output_dim
        decoder_embed_dim = decoder_input_embed_dim
        decoder_output_embed_dim = decoder_input_embed_dim

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DynamicConvDecoderLayer(decoder_embed_dim, decoder_conv_dim, decoder_glu,
                                    decoder_conv_type, weight_softmax, decoder_attention_heads,
                                    weight_dropout, dropout, relu_dropout, input_dropout,
                                    decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                                    article_embed_size, kernel_size=decoder_kernel_size_list[i])
            for i in range(decoder_layers)
        ])

        self.embed_out = nn.Parameter(torch.Tensor(vocab_size, decoder_output_embed_dim))
        # nn.init.normal_(self.embed_out, mean=0, std=decoder_output_embed_dim ** -0.5)

        self.register_buffer('version', torch.Tensor([2]))

        self.normalize = decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = nn.LayerNorm(decoder_embed_dim)

    def forward(self, prev_target, image, image_mask, article, article_mask,
                incremental_state=None, use_layers=None, **kwargs):

        # embed tokens and positions
        X_word_embed = self.word_embed(prev_target)
        X_pos_embed = self.pos_embed(prev_target, incremental_state=incremental_state)
        X = torch.stack([X_word_embed, X_pos_embed], dim=-1).sum(dim=-1)

        # if incremental_state is not None:
        #     X = X[:, -1:]

        X = F.dropout(X, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        X = X.transpose(0, 1)
        attn = None

        inner_states = [X]

        # decoder layers
        for i, layer in enumerate(self.layers):
            if not use_layers or i in use_layers:
                X, attn = layer(X, image, image_mask, article, article_mask, incremental_state,)
                inner_states.append(X)

        if self.normalize:
            X = self.layer_norm(X)

        # T x B x C -> B x T x C
        X = X.transpose(0, 1)

        return X, {'attn': attn, 'inner_states': inner_states}

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            target = sample['target'] if sample else None
            out = self.adaptive_softmax.get_log_prob(
                net_output[0], target)
            return out.exp() if not log_probs else out

        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @staticmethod
    def filter_incremental_state(incremental_state, active_idx):
        if incremental_state is None:
            return
        for key in incremental_state:
            if 'DynamicConv1dTBC' in key:
                incremental_state[key] = incremental_state[key][:, active_idx]


class DynamicConvDecoderLayer(nn.Module):
    def __init__(self, decoder_embed_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 article_embed_size, kernel_size=0):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.conv_dim = decoder_conv_dim
        if decoder_glu:
            self.linear1 = GehringLinear(self.embed_dim, 2*self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = GehringLinear(self.embed_dim, self.conv_dim)
            self.act = None
        if decoder_conv_type == 'lightweight':
            self.conv = LightweightConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                             weight_softmax=weight_softmax,
                                             num_heads=decoder_attention_heads,
                                             weight_dropout=weight_dropout)
        elif decoder_conv_type == 'dynamic':
            self.conv = DynamicConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=weight_softmax,
                                         num_heads=decoder_attention_heads,
                                         weight_dropout=weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = GehringLinear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.normalize_before = decoder_normalize_before

        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

        self.context_attns = nn.ModuleDict()
        self.context_attn_lns = nn.ModuleDict()
        C = 2048

        self.context_attns['image'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=C, vdim=C,
            dropout=attention_dropout)
        self.context_attn_lns['image'] = nn.LayerNorm(self.embed_dim)

        self.context_attns['article'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=article_embed_size, vdim=article_embed_size,
            dropout=attention_dropout)
        self.context_attn_lns['article'] = nn.LayerNorm(self.embed_dim)

        context_size = self.embed_dim * 2

        self.context_fc = GehringLinear(context_size, self.embed_dim)

        self.fc1 = GehringLinear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = GehringLinear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, X, image, image_mask, article, article_mask, incremental_state):
        residual = X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, before=True)
        X = F.dropout(X, p=self.input_dropout, training=self.training)
        X = self.linear1(X)
        if self.act is not None:
            X = self.act(X)
        X = self.conv(X, incremental_state=incremental_state)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, after=True)

        attn = None
        X_contexts = []

        # Image attention
        residual = X
        X_image = self.maybe_layer_norm(
            self.context_attn_lns['image'], X, before=True)
        X_image, attn = self.context_attns['image'](
            query=X_image,
            key=image,
            value=image,
            key_padding_mask=image_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_image = F.dropout(X_image, p=self.dropout, training=self.training)
        X_image = residual + X_image
        X_image = self.maybe_layer_norm(
            self.context_attn_lns['image'], X_image, after=True)
        X_contexts.append(X_image)

        # Article attention
        residual = X
        X_article = self.maybe_layer_norm(
            self.context_attn_lns['article'], X, before=True)
        X_article, attn = self.context_attns['article'](
            query=X_article,
            key=article,
            value=article,
            key_padding_mask=article_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_article = F.dropout(X_article, p=self.dropout,
                              training=self.training)
        X_article = residual + X_article
        X_article = self.maybe_layer_norm(
            self.context_attn_lns['article'], X_article, after=True)

        X_contexts.append(X_article)

        X_context = torch.cat(X_contexts, dim=-1)
        X = self.context_fc(X_context)

        residual = X
        X = self.maybe_layer_norm(self.final_layer_norm, X, before=True)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.relu_dropout, training=self.training)
        X = self.fc2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.final_layer_norm, X, after=True)
        return X, attn

    def maybe_layer_norm(self, layer_norm, X, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(X)
        else:
            return X

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.relu_dropout, self.input_dropout, self.normalize_before)
