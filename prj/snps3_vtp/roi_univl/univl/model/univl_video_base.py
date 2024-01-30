# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import copy

import torch
import torch.nn.functional as F
from torch import nn

from antmmf.modules.encoders import TextEncoder, VisualEncoder
from .univl_base import split_encoder_output


class UnivlVideoBase(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.arch_type = self.config.get("arch_type", "univl")
        self.with_cross_encoder = kwargs.get("with_cross_encoder", None)
        if self.with_cross_encoder is None:
            self.with_cross_encoder = self.config.with_cross_encoder
        self.build()

    def build(self):
        # text encoder & embedding ===>
        self.text_encoder = TextEncoder(self.config.text_encoder).module

        # image encoder & embedding
        self.img_encoder = VisualEncoder(self.config.image_encoder).module

        self.img_proj = None
        if self.img_encoder.out_dim != self.config.hidden_size:
            self.img_proj = nn.Parameter(
                torch.empty((self.img_encoder.out_dim, self.config.hidden_size))
            )

        if self.arch_type == "univl":
            self.img_encoder.add_module(
                "img_fc",
                nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                ),
            )

        self.cross_embeddings = self.text_encoder.embeddings
        self.cross_encoder = self.text_encoder.encoder

        # cross-modal encoder
        if self.with_cross_encoder is True and self.arch_type == "univl":
            # cross_encoder = TextEncoder(self.config.cross_encoder).module
            # cross_pooler不共享参数
            self.cross_pooler = copy.deepcopy(self.text_encoder.pooler)

    def forward_img_encoder(
        self,
        image_data,
        image_pad_mask,
        image_n_clips,
        image_num_frames,
        img_encoder=None,
        **kwargs
    ):
        img_encoder = img_encoder or self.img_encoder
        output_dict = img_encoder(image_data, image_mask=image_pad_mask)
        grid_feature = output_dict["grid_feature"]  # b, n_clips*n_frames, c, h, w
        grid_mask = output_dict["grid_mask"]  # b, n_clips*n_frames, h, w

        if self.img_proj is not None:  #
            grid_feature = torch.einsum(
                "bnchw, cj -> bnjhw", grid_feature, self.img_proj
            )

        # save grid feature size
        grid_shape = grid_feature.shape[-2:]

        # n_clips and n_frames should be same across batch
        n_clips, n_frames = image_n_clips[0], image_num_frames[0]
        bsz, c = grid_feature.size(0), grid_feature.size(2)

        # combine n_clips and batch_size dimension
        # ClipBERT ensembles clips at model level; ensembles frames at grid feature level
        grid_feature = grid_feature.contiguous().view(
            bsz * n_clips, n_frames, *grid_feature.shape[2:]
        )  # b*n_clips, n_frames, c, h, w
        grid_mask = grid_mask.contiguous().view(
            bsz * n_clips, n_frames, *grid_mask.shape[2:]
        )  # b*n_clips, n_frames, h, w

        # video clip feature as average pooling of frames
        # clip feature: b*n_clips, c
        g = grid_feature.transpose(1, 2).flatten(2)
        m = ~grid_mask.flatten(1).unsqueeze(1).expand_as(g)
        clip_feature = (g * m).sum(-1) / m.sum(-1)
        clip_tokens = clip_feature.view(bsz, n_clips, c)
        clip_mask = torch.zeros((bsz, n_clips), device=clip_tokens.device).bool()

        # # channel last to conform antmmf.modules.embeddings.ClipVisualEmbedding's input
        # grid_feature = grid_feature.permute(0, 1, 3, 4, 2)
        #
        # # remove temporal axis from now on
        # visual_embed, sampled_indices = self.img_embeddings(
        #     grid_feature
        # )  # [ b*n_clips, h*w, c]
        # # frames from the same clip should have same grid_mask
        # if grid_mask.size(1) > 1:  # num_frames > 1
        #     assert (grid_mask[:, 0] == grid_mask[:, 1]).all().item()
        # visual_mask = grid_mask[:, 0].view(bsz * n_clips, -1)
        # visual_mask = visual_mask.index_select(1, sampled_indices)
        if "img_fc" in img_encoder._modules:
            clip_feature = img_encoder.img_fc(clip_feature)

        clip_feature = F.normalize(clip_feature, p=2, dim=-1)

        return_dict = dict(
            visual_embed=clip_tokens,  # b, seq_length, c
            visual_mask=clip_mask,  # b, seq_length:  false indicating pixel areas
            visual_grid_shape=grid_shape,
            clip_feature=clip_feature,  # b*n_clips, c
        )
        return return_dict

    def forward_text_encoder(self, input_ids, input_mask, txt_encoder=None):
        token_type_ids = torch.zeros_like(input_ids)

        text_encoder = txt_encoder or self.text_encoder
        # BERT attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions
        if self.training:
            if self.arch_type == "univl":
                sequence_output, pooled_output, att = text_encoder(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                )
                # get important words
                words_importance = (
                    torch.cat([a.mean(1, keepdim=True) for a in att], dim=1)
                    .sum(dim=(1, 2))
                    .detach()
                )
            elif self.arch_type == "clip":
                sequence_output, pooled_output = text_encoder(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                )
                words_importance = None
        else:
            words_importance = None
            sequence_output, pooled_output = text_encoder(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=token_type_ids,
                output_attentions=False,
            )
        pooled_output = F.normalize(pooled_output, p=2, dim=-1)

        return_dict = dict(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            input_mask=input_mask,
            words_importance=words_importance,
        )
        return return_dict

    def prepare_cross_text(self, input_ids, input_mask):
        cap_mask = input_mask  # b_txt,
        txt_token_type_ids = torch.zeros_like(input_ids)
        cap_embed = self.cross_embeddings(
            input_ids=input_ids, token_type_ids=txt_token_type_ids
        )

        batch_size = cap_embed.shape[0]
        return cap_embed, cap_mask, batch_size

    def prepare_cross_visual(self, visual_embed, visual_mask=None):
        bsz, num_clip = visual_embed.shape[0], visual_embed.shape[1]
        if visual_mask is None:
            visual_mask = torch.zeros(
                (bsz, num_clip), device=visual_embed.device
            ).bool()

        input_shape = visual_embed.size()[:-1]
        SEP_ID = torch.zeros(
            input_shape[0], dtype=torch.long, device=visual_embed.device
        ).fill_(102)
        sep_token_embeds = self.cross_embeddings.word_embeddings(SEP_ID).unsqueeze(1)
        visual_input = visual_embed
        visual_mask = visual_mask.logical_not().long()
        visual_inputs_embeds = torch.cat([visual_input, sep_token_embeds], 1)
        token_type_ids = torch.ones(
            (visual_inputs_embeds.size(0), visual_inputs_embeds.size(1)),
            dtype=torch.long,
            device=visual_inputs_embeds.device,
        )
        new_visual_embed = self.cross_embeddings(
            inputs_embeds=visual_inputs_embeds, token_type_ids=token_type_ids
        )  # b, num_clips, c
        new_visual_mask = torch.cat(
            [visual_mask, visual_mask.new_ones((input_shape[0], 1))], 1
        )
        return new_visual_embed, new_visual_mask, num_clip

    def build_transformer_input(
        self, visual_embed_dict, text_embed_dict, caption_input
    ):
        cap_embed, cap_mask, batch_size = self.prepare_cross_text(
            caption_input["caption_input_ids"], caption_input["caption_input_mask"]
        )
        visual_embed, visual_mask, num_clip = self.prepare_cross_visual(
            visual_embed_dict["visual_embed"], visual_embed_dict["visual_mask"]
        )
        return (
            cap_embed,
            visual_embed,
            cap_mask,  # b_txt, seq, hidden,
            visual_mask,
            num_clip,
            batch_size,
        )

    def get_cross_output(self, cap_embed, visual_embed, cap_mask, visual_mask, n_clips):
        cap_embed, cap_mask = self._align_text_to_video_clips(
            cap_embed, cap_mask, n_clips
        )

        # cap_embed已包含[CLS] 和 [SEP] token
        embed = [cap_embed, visual_embed]
        mask = [cap_mask, visual_mask]

        # modality range info
        position_range = torch.cumsum(
            torch.tensor([x.size(1) for x in embed[:-1]]), dim=0
        )

        embed = torch.cat(embed, 1)
        mask = torch.cat(mask, 1)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.cross_encoder(
            embed,
            attention_mask=extended_attention_mask,
            head_mask=[None] * len(self.cross_encoder.layer),
        )
        sequence_output = encoder_outputs[0]
        if self.arch_type == "univl":
            pooled_output = self.cross_pooler(sequence_output)
        elif self.arch_type == "clip":
            if self.text_encoder.text_projection is not None:
                pooled_output = (
                    sequence_output[:, 0, :] @ self.text_encoder.text_projection
                )
            else:
                pooled_output = sequence_output[:, 0, :]

        cap_seq_output, visual_seq_output = split_encoder_output(
            sequence_output, position_range
        )
        # remove SEP token for visual
        return (
            cap_seq_output,
            visual_seq_output[
                :,
                :-1,
            ],
            pooled_output,
        )

    def get_l2_input(self, img_input, caption_input):
        visual_embed_dict = self.forward_img_encoder(**img_input)

        text_embed_dict = self.forward_text_encoder(
            caption_input["caption_raw_input_ids"], caption_input["caption_input_mask"]
        )
        (
            cap_embed,
            visual_embed,
            cap_mask,
            visual_mask,
            num_clips,
            batch_size,
        ) = self.build_transformer_input(
            visual_embed_dict, text_embed_dict, caption_input
        )
        text_embed_l1, video_embed_l1 = (
            text_embed_dict["pooled_output"],  # bsz, hidden
            visual_embed_dict["clip_feature"],  # bsz * n_clips, hidden
        )
        cap_input = (
            cap_embed,
            cap_mask,
            text_embed_l1,
            batch_size,
        )
        vis_input = (visual_embed, visual_mask, video_embed_l1, num_clips)
        return cap_input, vis_input, text_embed_dict, visual_embed_dict

    def _align_text_to_video_clips(self, cap_embed, cap_mask, num_clip: int = 1):
        if num_clip > 1:
            # replicate text output to cover multiple time clips in video
            cap_embed = cap_embed.repeat_interleave(num_clip, dim=0)
            cap_mask = cap_mask.repeat_interleave(num_clip, dim=0)
        return cap_embed, cap_mask

    def forward(self, img_input, caption_input):
        cap_input, vis_input, _, _ = self.get_l2_input(img_input, caption_input)
        (cap_embed, cap_mask, text_embed_l1, batch_size) = cap_input
        (visual_embed, visual_mask, video_embed_l1, num_clips) = vis_input
        cap_seq_output, visual_seq_output, pooled_output = self.get_cross_output(
            cap_embed, visual_embed, cap_mask, visual_mask, num_clips
        )
        return cap_seq_output, visual_seq_output, pooled_output
