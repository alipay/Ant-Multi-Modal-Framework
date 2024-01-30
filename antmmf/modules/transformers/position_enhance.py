# coding: utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

from typing import Any, Dict, Union
from typing import Optional

import torch
from dataclasses import field
from torch import nn

from antmmf.common import configurable, Configuration


class PositionEnhancedTransformer(nn.Module):
    """
    It is inspired from Detr experimental results in Table. 3.

    PositionEnhancedTransformer passes  positional encodings directly in attention
    at each layer(for both encoder & decoder), which results in an position enhanced
    transformer.

    We use fixed sine position embedding for src and learned embedding for targets.
    """

    @configurable
    def __init__(
        self,
        encoder_config: Union[Dict[str, Any], Configuration] = field(
            default_factory=dict
        ),
        decoder_config: Union[Dict[str, Any], Configuration] = None,
        num_queries: int = 100,
        decoding_type: str = "generation",
    ):
        super().__init__()
        from antmmf.modules.encoders import MultimodalEncoder
        from antmmf.modules.build import build_decoder

        self.encoder = MultimodalEncoder(encoder_config).module
        if decoder_config is not None:
            self.decoder = build_decoder(decoder_config)
            self.decoder_d_model = decoder_config["params"]["params"]["d_model"]

            # initialize task related modules
            if "detr" in decoding_type:
                self.query_embed = nn.Embedding(num_queries, self.decoder_d_model)

            if "generation" in decoding_type:
                # if query dimension doesn't match the multimodal transformer decoder hidden dimension,
                # add a linear projection layer between the two
                gen_query_dim = decoder_config["params"]["params"].get(
                    "generation_query_dim", None
                )
                if (
                    gen_query_dim is not None
                    and gen_query_dim != decoder_config["params"]["params"]["d_model"]
                ):
                    self.gen_query_linear = nn.Linear(
                        gen_query_dim, self.decoder_d_model
                    )
                else:
                    self.gen_query_linear = nn.Identity()

        self.decoding_type = decoding_type
        self.decoder_config = decoder_config
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_encoder(
        self, src, src_pos=None, src_mask=None, src_key_padding_mask=None, **kwargs
    ):
        return self.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
            pos=src_pos,
            mask=src_mask,
            **kwargs,
        )

    def forward_decoder(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        decoding_type = kwargs.pop("decoding_type", None) or self.decoding_type
        if decoding_type == "detr":
            # query position embedding(learned)
            if query_pos is None:
                query_pos = self.query_embed.weight  # query_seq_length, hidden
            query_pos = query_pos.unsqueeze(1).repeat(
                1, memory.size(1), 1
            )  # query_seq_length, bsz, hidden
            detr_res = self._detr_decode(
                tgt, memory, memory_key_padding_mask, memory_pos, query_pos
            )
            decoder_output = {decoding_type: detr_res}
        elif decoding_type == "generation":
            generation_res = self._regressive_decode(
                tgt, memory, memory_key_padding_mask, memory_pos, query_pos, **kwargs
            )
            decoder_output = {decoding_type: generation_res}
        elif "+" in decoding_type:
            decoder_output = dict()
            for _type in decoding_type.split("+"):
                decoder_out = self.forward_decoder(
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    memory_pos,
                    query_pos,
                    decoding_type=_type,
                    **kwargs,
                )
                decoder_output.update(decoder_out)
        else:
            raise Exception(f"unknow decoding_type:{self.decoding_type}")

        return decoder_output

    def forward(
        self,
        src,
        src_key_padding_mask,
        memory_key_padding_mask,
        query_embed=None,
        src_pos=None,
        tgt=None,
        **kwargs,
    ):
        """
        :param src(torch.float32): (b,seq_length, hidden_dim)
        :param src_key_padding_mask(torch.bool): (b, seq_length)
        :param memory_key_padding_mask(torch.bool): (b, seq_length),
        :param query_embed(torch.float32): (query_seq_length, hidden_dim)
        :param src_pos(torch.float32): (b, seq_length, hidden_dim), position embedding enhanced at attention.
        :param tgt(torch.float32): (b, query_seq_length, hidden)

        Note: For src_key_padding_mask & memory_key_padding_mask, When given a binary mask and a value is True,
        the corresponding value on the attention layer will be ignored.

        :return:
            memory: Encoder output, [bs, src_length, hidden]
            hs: Decoder output, [#decoder_layers, bs, query_seq_length, hidden_dim]
        """
        bsz, seq_len, hidden = src.shape
        src = src.transpose(0, 1)  # seq_length, bsz, hidden
        if src_pos is not None:
            src_pos = src_pos.transpose(0, 1)  # seq_length, bsz, hidden

        # conform to pytorch transformer's data_format
        # src & pos_embeded: bsz, seq_len, hidden -> seq_len, bsz, hidden
        memory = self.forward_encoder(
            src, src_pos=src_pos, src_key_padding_mask=src_key_padding_mask
        )
        decoder_output = dict()
        if self.decoder_config is not None:
            decoder_output = self.forward_decoder(
                tgt,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=src_pos,
                query_pos=query_embed,
                **kwargs,
            )

        output = dict(
            decoder=decoder_output,
            memory=memory.permute(1, 0, 2).reshape(bsz, seq_len, hidden),
        )
        return output

    def _detr_decode(
        self, tgt, memory, memory_key_padding_mask, memory_pos, query_pos, **kwargs
    ):
        # Note: tgt is not used for detr decode
        tgt = torch.zeros_like(query_pos)  # query token
        # parallel decoding: https://github.com/facebookresearch/detr/issues/3
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=memory_pos,
            query_pos=query_pos,
        )
        if hs.ndim == 3:
            hs = hs.transpose(0, 1)  # b, query_length, hidden
        else:  # with hier decoder layer output
            assert hs.ndim == 4
            hs = hs.transpose(1, 2)  # #num_decoder_layer, b, query_length, hidden
        return hs

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _regressive_decode(
        self,
        tgt,
        memory,
        memory_key_padding_mask,
        memory_pos,
        query_pos=None,
        caption_output: Dict[str, Any] = None,
        text_embedding: nn.Module = None,
        tokenizer=None,
        generation_head=None,
        dec_step_num: int = 20,
    ):
        assert generation_head is not None
        return_dict = dict(
            generation_ids=None, generation_text=None, generation_logits=None
        )
        if caption_output is not None:
            # regressive decode by masking words in a parallel way for training
            # https://github.com/microsoft/UniVL/blob/main/modules/module_decoder.py#L393-L396
            # caption_ids: SOS: CLS-101, EOS: SEP-102
            caption_ids = caption_output["generation_input_ids"]

            # Note: decoder inputs embedding should not have token type embedding, however this
            # introduces token type embedding for simplicity.
            tgt = text_embedding(input_ids=caption_ids)  # bsz, seq_length, hidden

            # mapping to d_model
            tgt = self.gen_query_linear(tgt)

            tgt = tgt.transpose(
                0, 1
            )  # convert to decoder #seq_length, bsz, hidden format

            # ensure only earlier tokens can be seen for decoder.
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

            # for caption generation task, not use learned query_pos.
            hs = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=None,  # no mask for memory
                tgt_key_padding_mask=None,  # query is not padded
                memory_key_padding_mask=memory_key_padding_mask,  # skip memory padding
                pos=memory_pos,
                query_pos=None,
            )

            if hs.ndim == 3:
                hs = hs.transpose(0, 1)  # b, query_length, hidden
            else:  # with hier decoder layer output
                assert hs.ndim == 4
                hs = hs.transpose(1, 2)  # #num_decoder_layer, b, query_length, hidden
                hs = hs[-1]  # only get last layer for generation

            generation_logits = generation_head.forward_head(encoder_output=hs)
            generation_ids = generation_logits.argmax(dim=-1)
            generation_text = generation_head.convert_id2text(tokenizer, generation_ids)
            return_dict["generation_logits"] = generation_logits
            return_dict["generation_ids"] = generation_ids
            return_dict["generation_text"] = generation_text

        else:
            # Generation Reference:
            # offical code: https://github.com/pytorch/tutorials/issues/719#issuecomment-798983859
            # greedy decode: https://github.com/microsoft/TAP/blob/\
            #                352891f93c75ac5d6b9ba141bbe831477dcdd807/pythia/models/tap.py#L343-L364
            # beam decode: https://github.com/microsoft/UniVL/blob/\
            #                1a40788874460e1f17691e749af7951d0e872523/main_task_caption.py#L550-L575
            bsz, device = memory.size(1), memory.device
            input_ids = memory.new_zeros([bsz, dec_step_num + 1], dtype=torch.long)
            input_ids[:, 0] = tokenizer.cls_token_id
            for i in range(dec_step_num):
                tgt = text_embedding(input_ids=input_ids)[:, : i + 1]
                # mapping to d_model
                tgt = self.gen_query_linear(tgt)
                # convert to decoder #seq_length, bsz, hidden format
                tgt = tgt.transpose(0, 1)
                # ensure only earlier tokens can be seen for decoder.
                tgt_mask = self.generate_square_subsequent_mask(i + 1).to(device)
                hs = self.decoder(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,  # no mask for memory
                    tgt_key_padding_mask=None,  # query is not padded
                    memory_key_padding_mask=memory_key_padding_mask,  # skip memory padding
                    pos=memory_pos,
                    query_pos=None,
                )  # num_decoder_layer, query_length, b,  hidden

                # the last timestep of last decoder layer
                cur_hs = hs[-1][-1].unsqueeze(1)  # bsz, 1, hidden,
                cur_logits = generation_head.forward_head(
                    encoder_output=cur_hs
                )  # bsz, 1, vocab_size

                # greedy decoding at test time
                indices = cur_logits.argmax(dim=-1)
                input_ids[:, i + 1] = indices.squeeze(-1).squeeze(-1)

            generation_ids = input_ids[:, 1:]
            generation_text = generation_head.convert_id2text(tokenizer, generation_ids)
            return_dict["generation_ids"] = generation_ids
            return_dict["generation_text"] = generation_text

        return return_dict
