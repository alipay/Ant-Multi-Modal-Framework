# coding=utf-8
# Copyright (c) 2023 Ant Group and its affiliates.

import logging
import torch
from torch import nn
import torch.nn.functional as F
from antmmf.utils.distributed_utils import gather_tensor
from antmmf.modules.vision.backbone.clip.model import QuickGELU
from collections import OrderedDict
from .tpmcl_utils import LinearXWeightPredictor, AttentionXWeightPredictor, TokenImportanceSelector
logger = logging.getLogger(__name__)

class DmaeUtils(nn.Module):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self.interaction = getattr(self.config, "l3_interaction", "wti")
        self.with_va = getattr(self.config, "l3_with_nfc", True)
        self.wti_arch = getattr(self.config, "l3_wti_arch", 1)
        self.sim_header = getattr(self.config, "l3_sim_header", "meanP")
        self.partial_type = getattr(self.config, "l3_partial_type", 4)
        self.max_frames = getattr(self.config, "l3_max_frames", 8)
        self.max_words = getattr(self.config, "l3_max_words", 30)
        # context_length = self.config.img_embeddings.params.max_position_embeddings
        self.cross_num_hidden_layers = getattr(self.config, "l3_sim_header_hidden_layer", 4)
        hidden_size = getattr(self.config, "hidden_size", 768)
        assert self.sim_header in ["meanP", "seqTransf"]
        self.init_tpmcl = True
        if self.partial_type > 0:
            self._run_init_tpmcl()
        assert self.sim_header in ["meanP", "seqTransf"]

        transformer_width = hidden_size
        if "wti" in self.interaction:  # self.interaction == "wti":
            if self.wti_arch == 1:
                self.text_weight_fc = nn.Linear(transformer_width, 1)
                self.video_weight_fc = nn.Linear(transformer_width, 1)
            elif self.wti_arch == 2:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            elif self.wti_arch == 3:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
        if self.sim_header == "seqTransf":
            context_length = 77
            max_position_embeddings = context_length  # 77
            transformer_heads = transformer_width // 64
            self.frame_position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
            self.transformerClip = TransformerClip(width=transformer_width,
                                                   layers=self.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
            # self.frame_position_embeddings = nn.Embedding(600, cross_config.hidden_size)

    def _run_init_tpmcl(self):
        embed_dim = self.config.hidden_size
        max_frames = self.max_frames + 1
        self.xwp_type = 'linear'
        if self.xwp_type == 'linear':
            self.t2v_linear_xwp = LinearXWeightPredictor(num_frames=1, num_tokens=max_frames, embed_dim=embed_dim)
            self.v2t_linear_xwp = LinearXWeightPredictor(num_frames=max_frames, num_tokens=self.max_words,
                                                         embed_dim=embed_dim)
        else:
            self.t2v_attention_xwp = AttentionXWeightPredictor(num_frames=1, num_tokens=max_frames,
                                                               embed_dim=embed_dim)
            self.v2t_attention_xwp = AttentionXWeightPredictor(num_frames=max_frames, num_tokens=self.max_words,
                                                               embed_dim=embed_dim)
        cis_thresh = getattr(self.config, "l3_cis_thresh", 0.6)
        self.tis_selector = TokenImportanceSelector(cis_thresh)
        # Partial Mask
        margin_loss_thresh = getattr(self.config, "l3_margin_loss_thresh", 0.6)
        self.margin_loss_fct = torch.nn.MarginRankingLoss(margin=margin_loss_thresh, reduction='mean')
        self.init_tpmcl = False

    def _get_wti_similarity(self, text_feat, video_feat, text_mask, video_mask,
                            text_weight=None, video_weight=None, self_weight=False):
        text_sum = text_mask.sum(-1).unsqueeze(1)
        video_sum = video_mask.sum(-1).unsqueeze(1)
        text_att = text_mask / (text_mask.sum(-1).unsqueeze(1))
        video_att = video_mask / (video_mask.sum(-1).unsqueeze(1))
        text_mask = torch.clamp(text_mask, 0, 1)
        # get sim: [bt, bv, max_words, max_frames]
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        if self_weight:
            self_frames = torch.einsum('atd,avd->atv', [video_feat, video_feat])
            self_frames = torch.einsum('atv,at->atv', [self_frames, video_mask])
            self_frames = torch.einsum('atv,av->atv', [self_frames, video_mask])  # [bs,bs,max_frames,max_frames]
            self_frames[:, torch.arange(self_frames.shape[1]), torch.arange(self_frames.shape[1])] = 0.
            f2f_logits, max_f2f_idx1 = self_frames.max(dim=-1)
            # select max_2nd_frame_sim
            xindex = torch.arange(max_idx1.shape[0]).repeat_interleave(max_idx1.shape[1])
            yindex = torch.repeat_interleave(torch.arange(max_idx1.shape[1]).unsqueeze(-1), max_idx1.shape[0],
                                             dim=1).permute(1, 0).flatten()
            zindex = max_idx1[xindex, yindex]  # [bs*bs,max_words]
            xindex_full = xindex.repeat(zindex.shape[-1])
            yindex_full = yindex.repeat(zindex.shape[-1])  # [bs*bs*max_words]
            zindex_full = zindex.permute(1, 0).flatten()
            zindex2_full = max_f2f_idx1[yindex_full, zindex_full]
            att_self_weight = f2f_logits[yindex_full, zindex_full]
            zi_full = torch.arange(zindex.shape[-1]).repeat_interleave(zindex.shape[0])
            att_max_sim = retrieve_logits[xindex_full, yindex_full, zi_full, zindex2_full]
            att_max_sim = att_self_weight * att_max_sim
            att_max_sim = att_max_sim.view(-1, t2v_logits.shape[0], t2v_logits.shape[1]).permute(1, 2, 0)
            # add max_2nd_frame_sim to max_1st_frame_sim
            t2v_logits = t2v_logits + att_max_sim * 0.5
        # max for video token
        if "wti" in self.interaction:  # if self.interaction == 'wti':  # weighted token-wise interaction
            t2v_logits = torch.einsum('abt,at->abt', [t2v_logits, text_att * text_sum])
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
            v2t_logits = torch.einsum('abv,bv->abv', [v2t_logits, video_att * video_sum])
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        else:  # if self.interaction == 'ti':  # token-wise interaction
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_att])
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_att])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        return retrieve_logits

    def wti_interaction(self, text_feat, word_feat,
                        video_feat, word_mask, video_mask):
        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat = gather_tensor(text_feat, method="cat", back_gradient=True, pad_tensors=True)
            if word_feat is not None:
                word_feat = gather_tensor(word_feat, method="cat", back_gradient=True,
                                          pad_tensors=True)  # [bt,max_words, hidden]
            video_feat = gather_tensor(video_feat, method="cat", back_gradient=True,
                                       pad_tensors=True)  # [bv, max_frames, hidden]
            word_mask = gather_tensor(word_mask, method="cat", back_gradient=True,
                                      pad_tensors=True)  # [bt, max_words]
            video_mask = gather_tensor(video_mask, method="cat", back_gradient=True,
                                       pad_tensors=True)  # [bv, max_frames]
            torch.distributed.barrier()  # force sync
        #########################################################
        # video mask need to change due to more tokens in frame #
        #########################################################
        expand_times = video_feat.shape[1] // video_mask.shape[1]
        # video_mask shape here is (bs, max_frames)
        video_mask = video_mask.unsqueeze(1).repeat(1, 1, expand_times).view(video_mask.shape[0], -1)
        #########################################################
        # video mask need to change due to more tokens in frame #
        #########################################################
        #########################################################
        # text mask need to merge due to less tokens in sentence #
        #########################################################
        # word_mask = text_mask
        if word_mask.shape[1] != text_feat.shape[1]:
            text_mask = word_mask[:, 0].view(text_feat.shape[0], -1)
        # if self.interaction == 'wti':
        text_weight, word_weight, video_weight = None, None, None
        if "wti" in self.interaction:
            text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
            if word_feat is not None:
                word_weight = self.text_weight_fc(word_feat).squeeze(2)  # B x N_t x D -> B x N_t
                word_weight.masked_fill_(~(torch.tensor(word_mask, dtype=torch.bool)), float("-inf"))
                word_weight = torch.softmax(word_weight, dim=-1)  # B x N_t
            video_weight = self.video_weight_fc(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
            video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
            video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
        # get sentence2frames sim: [bt, bv]
        sim_matrix_sent2frames = self._get_wti_similarity(text_feat, video_feat, text_mask, video_mask, text_weight,
                                                          video_weight, self_weight=self.with_va)
        # get words2frames sim: [bt, bv]
        retrieve_logits = sim_matrix_sent2frames
        if self.interaction in ["att_ti", "att_wti"] and word_feat is not None:
            sim_matrix_word2frames = self._get_wti_similarity(word_feat, video_feat, word_mask, video_mask,
                                                              word_weight, video_weight, self_weight=self.with_va)
            retrieve_logits = (sim_matrix_sent2frames + sim_matrix_word2frames) / 2.0
        return retrieve_logits

    def _agg_visual_feat(self, visual_output, video_mask, sim_header="meanP"):
        #########################################################
        # video mask need to change due to more tokens in frame #
        # #########################################################
        expand_times = visual_output.shape[1] // video_mask.shape[1]
        # # video_mask shape here is (bs, max_frames)
        video_token_mask = video_mask.unsqueeze(1).repeat(1, 1, expand_times).view(video_mask.shape[0], -1)
        # #########################################################
        # # video mask need to change due to more tokens in frame #
        # #########################################################
        #
        if sim_header == "meanP":
            # Default: Parameter-free type
            visual_output_original = visual_output
            pass
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings
            extended_video_mask = (1.0 - video_token_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_token_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            # consider remove below statement because it seems non-sense......(cannot, performance decay)
            visual_output = visual_output + visual_output_original
        #########################################################
        # video mask need to change due to more tokens in frame #
        #########################################################
        frame_embedding_index = torch.arange(start=0, end=visual_output.shape[1], step=expand_times, dtype=torch.long,
                                             device=visual_output.device)
        visual_output = visual_output[:, frame_embedding_index, :]
        visual_output_original = visual_output_original[:, frame_embedding_index, :]
        video_token_mask = video_token_mask[:, frame_embedding_index]
        #########################################################
        # video mask need to change due to more tokens in frame #
        #########################################################
        return visual_output, video_token_mask, visual_output_original

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        if isinstance(sequence_output, tuple):
            sequence_token_hidden = sequence_output[1]  # cap_embed/word_feat
            sequence_output = sequence_output[0]  # text_embed_l1
        else:
            sequence_token_hidden = None
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if sequence_token_hidden is not None:
            sequence_token_hidden = sequence_token_hidden.contiguous()
        loss = 0.
        agg_visual_output, video_token_mask, visual_output_original = self._agg_visual_feat(visual_output, video_mask,
                                                                                            sim_header=sim_header)
        # if self.interaction in ['wti','ti']:
        if "ti" in self.interaction:  # ["wti", "ti"]
            sim_matrix_semantic = self.wti_interaction(sequence_output, sequence_token_hidden, agg_visual_output,
                                                       attention_mask, video_token_mask)
        else:
            raise NotImplementedError
        return sim_matrix_semantic

    def get_similarity_logits(self, vis_input, cap_input, output_dict=None, shaped=False, loose_type=False):
        """
         :return:
             bsz_text, bsz_video, num_clips: clip-level score for training
             bsz_text, num_clips: clip-level score for val/test
         """
        (cap_embed, cap_mask, text_embed_l1, batch_size, twm_cap_mask, caption_input) = cap_input
        (visual_embed, visual_mask, video_embed_l1, num_clips, img_input) = vis_input
        cap_embed = cap_embed / cap_embed.norm(dim=-1, keepdim=True)
        visual_embed = visual_embed / visual_embed.norm(dim=-1, keepdim=True)
        if twm_cap_mask is not None:
            cap_mask = twm_cap_mask
        if shaped is False:
            cap_mask = cap_mask.view(-1, cap_mask.shape[-1])
            visual_mask = visual_mask.view(-1, visual_mask.shape[-1])
        text_embed_l1 = text_embed_l1.unsqueeze(1)

        if loose_type:
            #  assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            cap_output = (text_embed_l1, cap_embed)
            simi_matrix = self._loose_similarity(cap_output, visual_embed,
                                                 cap_mask, visual_mask, sim_header=self.sim_header)
            margin_loss = 0.
            if self.training and self.partial_type > 0:
                margin_loss = self.get_partial_similarity(cap_output, visual_embed,
                                                          cap_mask, visual_mask, self.partial_type)
        else:
            #   assert self.sim_header in ["tightTransf"]
            pass
        return simi_matrix, margin_loss

    def get_partial_similarity(self, sequence_output, visual_output, attention_mask, video_mask, partial_type=1):
        """
            Args:
           #     sim_matrix_semantic: [bt, bv]
                sequence_output: sequence tuple, (sentence-level, word-level feats)
                visual_output:  visual feats, (frame-level)
                attention_mask: sequence mask, (word-level)
                video_mask: visual mask, (frame-level)
                partial_type: =1 sim_global;
                            =2 sim_global + margin_loss1;
                            =3 sim_global + margin_loss2;
                            =4 sim_global+margin_loss1 + margin_loss2
        """
        # input feature tuple flatten
        if isinstance(sequence_output, tuple):
            sequence_token_feats = sequence_output[1]  # cap_embed/word_feat
            sequence_output = sequence_output[0]  # text_embed_l1
        else:
            sequence_token_feats = None
        bs_text, num_text, embed_dim = sequence_output.size()  # [bs, num_tokens, embed_dim]
        bs_visual, num_visual, embed_dim = visual_output.size()  # [bs, num_frames, embed_dim]
        step_size = 8
        split_size = [step_size] * (bs_text // step_size)
        res_size = bs_text - sum(split_size)
        if res_size > 0:
            split_size += [res_size]
        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        sequence_tokens_feats_splits = torch.split(sequence_token_feats, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        vis_step_size = 16
        vis_split_size = [vis_step_size] * (bs_visual // vis_step_size)
        vis_res_size = bs_visual - sum(vis_split_size)
        if vis_res_size > 0:
            vis_split_size += [vis_res_size]
        visual_output_splits = torch.split(visual_output, vis_split_size, dim=0)
        video_mask_splits = torch.split(video_mask, vis_split_size, dim=0)
        # sim global and partial sim matrix
        sim_global, t2vh_sim, t2vhh_sim, tg2vh_sim, tg2vhh_sim, tgh2vh_sim = [], [], [], [], [], []
        for i in range(len(split_size)):
            sequence_output_block = sequence_output_splits[i]
            sequence_tokens_feats_block = sequence_tokens_feats_splits[i]
            attention_mask_block = attention_mask_splits[i]
            sequence_output_block = (sequence_output_block, sequence_tokens_feats_block)
            each_row_sim, each_row_t2vh, each_row_t2vhh = [], [], []
            each_row_tg2vh, each_row_tg2vhh, each_row_tgh2vh = [], [], []
            for j in range(len(vis_split_size)):
                visual_output_block = visual_output_splits[j]
                video_mask_block = video_mask_splits[j]
                visual_output_block = (visual_output_block)
                visual_mask_block = (video_mask_block)
                t2vh, t2vhh, tg2vh, tg2vhh, tgh2vh, t2vg, t2vgh = self._get_partial_output(sequence_output_block,
                                                                                           visual_output_block,
                                                                                           attention_mask_block,
                                                                                           visual_mask_block,
                                                                                           xwp_type="linear",
                                                                                           partial_type=partial_type)
                each_row_t2vh.append(t2vh)
                each_row_t2vhh.append(t2vhh)
                each_row_tg2vh.append(tg2vh)
                each_row_tg2vhh.append(tg2vhh)
                each_row_tgh2vh.append(tgh2vh)
            if self.training and partial_type >= 2:
                each_row_t2vh = torch.cat(each_row_t2vh, dim=-1)
                each_row_t2vhh = torch.cat(each_row_t2vhh, dim=-1)
                each_row_tg2vh = torch.cat(each_row_tg2vh, dim=-1)
                each_row_tg2vhh = torch.cat(each_row_tg2vhh, dim=-1)
                each_row_tgh2vh = torch.cat(each_row_tgh2vh, dim=-1)
                t2vh_sim.append(each_row_t2vh)
                t2vhh_sim.append(each_row_t2vhh)
                tg2vh_sim.append(each_row_tg2vh)
                tg2vhh_sim.append(each_row_tg2vhh)
                tgh2vh_sim.append(each_row_tgh2vh)
        # margin loss
        margin_loss = 0.
        if self.training and partial_type >= 2:
            t2vh_sim = torch.cat(t2vh_sim, dim=0)
            t2vhh_sim = torch.cat(t2vhh_sim, dim=0)
            tg2vh_sim = torch.cat(tg2vh_sim, dim=0)
            tg2vhh_sim = torch.cat(tg2vhh_sim, dim=0)
            tgh2vh_sim = torch.cat(tgh2vh_sim, dim=0)
            # margin loss over sim matrix in batch
            if torch.cuda.is_available():
                t2vh_sim = gather_tensor(t2vh_sim, method="cat", back_gradient=True, pad_tensors=True)
                t2vhh_sim = gather_tensor(t2vhh_sim, method="cat", back_gradient=True, pad_tensors=True)
                tg2vh_sim = gather_tensor(tg2vh_sim, method="cat", back_gradient=True, pad_tensors=True)
                tg2vhh_sim = gather_tensor(tg2vhh_sim, method="cat", back_gradient=True, pad_tensors=True)
                tgh2vh_sim = gather_tensor(tgh2vh_sim, method="cat", back_gradient=True, pad_tensors=True)
                torch.distributed.barrier()  # force sync
            if partial_type == 2:  # margin1 + margin2
                margin_loss += self._get_partial_loss(t2vh_sim, t2vhh_sim)
                margin_loss += self._get_partial_loss(tg2vh_sim, tg2vhh_sim)
            if partial_type == 3:  # margin3
                margin_loss += self._get_partial_loss(tg2vh_sim, tgh2vh_sim)
            if partial_type == 4:  # margin1 + margin2 +margin3
                margin_loss += self._get_partial_loss(t2vh_sim, t2vhh_sim)
                margin_loss += self._get_partial_loss(tg2vh_sim, tg2vhh_sim)
                margin_loss += self._get_partial_loss(tg2vh_sim, tgh2vh_sim)
        return margin_loss

    def _get_partial_loss(self, sim_matrix, sim_matrix_bar):
        bt_anchor, bv_anchor = sim_matrix.size()
        bt, bv = sim_matrix_bar.size()
        if bv_anchor != bv:
            expand_times = bv_anchor // bv
            sim_matrix_bar = sim_matrix_bar.repeat(1, expand_times)
        sim_matrix_diag = torch.diag(sim_matrix)
        sim_matrix_bar_diag = torch.diag(sim_matrix_bar)
        target = torch.ones_like(sim_matrix_diag, device=sim_matrix_diag.device, requires_grad=False)
        return self.margin_loss_fct(sim_matrix_diag, sim_matrix_bar_diag, target)

    def _get_partial_output(self, sequence_output, visual_output, attention_mask, video_mask,
                            xwp_type="linear", partial_type=-1):
        # input feature tuple flatten
        sequence_output, sequence_token_hidden = sequence_output
        bt, num_text, embed_dim = sequence_output.size()  # [bs_text, num_tokens, embed_dim]
        bv, num_visual, embed_dim = visual_output.size()  # [bs_visual, num_frames, embed_dim]
        # text feature flatten
        sequence_output_batch = sequence_output.repeat(bv, 1, 1)  # repeat by batch block
        sequence_token_hidden_batch = sequence_token_hidden.repeat(bv, 1, 1)
        sequence_token_hidden_row = sequence_token_hidden.repeat_interleave(bv, 0)
        attention_mask_batch = attention_mask.repeat(bv, 1)
        attention_mask_row = attention_mask.repeat_interleave(bv, 0)
        # visual feature flatten
        visual_output_batch = visual_output.repeat(bt, 1, 1)
        visual_output_row = visual_output.repeat_interleave(bt, 0)
        video_mask_batch = video_mask.repeat(bt, 1)
        video_mask_row = video_mask.repeat_interleave(bt, 0)
        if xwp_type == "linear":
            t_token_w = self.v2t_linear_xwp(visual_output_batch, sequence_token_hidden_row)  # [bt*bv, num_tokens]
            v_token_w = self.t2v_linear_xwp(sequence_output_batch, visual_output_row)  # [bt*bv, num_frames*(top_k+1)]
        else:
            t_token_w = self.v2t_attention_xwp(visual_output_batch, sequence_token_hidden_row)
            v_token_w = self.t2v_attention_xwp(sequence_output_batch, visual_output_row)
        # sim global by weight global
        sequence_global = torch.einsum('abd,ab->ad', sequence_token_hidden_row, t_token_w)
        sequence_global = sequence_global.reshape(bt, bv, embed_dim)  # [bt, bv, embed_dim]
        visual_global = torch.einsum('abd,ab->ad', visual_output_row, v_token_w)
        visual_global = visual_global.reshape(bv, bt, embed_dim).permute(1, 0, 2).reshape(bt, bv, embed_dim)
        sequence_global = sequence_global / sequence_global.norm(dim=-1, keepdim=True)
        visual_global = visual_global / visual_global.norm(dim=-1, keepdim=True)
        t2vh_sim, t2vhh_sim, tg2vh_sim, tg2vhh_sim, tgh2vh_sim = None, None, None, None, None
        t2vg_sim, t2vgh_sim = None, None
        if self.training and partial_type >= 2:
            sequence_global = sequence_global.reshape(-1, num_text, embed_dim)  # [bt*bv, 1, embed_dim]
            visual_global = visual_global.reshape(-1, num_text, embed_dim)  # [bt*bv, 1, embed_dim]
            sequence_token_feats_masked, _ = self.tis_selector(sequence_token_hidden_row, t_token_w)
            sequence_global_partial = torch.einsum('abd,ab->ad', sequence_token_feats_masked, t_token_w)
            sequence_global_partial = sequence_global_partial.reshape(-1, num_text, embed_dim)  # [bt*bv, 1, embed_dim]
            visual_token_feats_masked, _ = self.tis_selector(visual_output_row, v_token_w)
            visual_global_partial = torch.einsum('abd,ab->ad', visual_token_feats_masked, v_token_w)
            visual_global_partial = visual_global_partial.reshape(-1, num_text, embed_dim)  # [bt*bv, 1, embed_dim]
            # visual_token_hidden_partial = self.visual_token_selector(visual_token_feats_masked)
            visual_token_hidden_partial, video_mask_row, _ = self._agg_visual_feat(visual_token_feats_masked,
                                                                                   video_mask_row,
                                                                                   sim_header=self.sim_header)
            # partial sim1a  (t_cls,v_tokens')
            t2vhh_sim = self._loose_similarity_row(sequence_output_batch, visual_token_hidden_partial,
                                                   attention_mask_batch, video_mask_row, sim_header=self.sim_header)
            t2vhh_sim = t2vhh_sim.reshape(bv, bt).T  # [t1v1, t2v1, tnv1,...,t1vn,t2vn,tnvn]
            # positive1  (t_cls, v_tokens) -> (t_cls, v_tokens')
            t2vh_sim = self._loose_similarity_row(sequence_output_batch, visual_output_batch, attention_mask_batch,
                                                  video_mask_batch, sim_header=self.sim_header)
            t2vh_sim = t2vh_sim.reshape(bt, bv)
            # positive2  (t_g, v_tokens)  ->  (t_g', v_tokens), (t_g, v_tokens')
            tg2vh_sim = self._loose_similarity_row(sequence_global, visual_output_batch, attention_mask_row,
                                                   video_mask_batch, sim_header=self.sim_header)
            tg2vh_sim = tg2vh_sim.reshape(bt, bv)  # [t1v1, t1v2, t1vn,...,tnv1,tnv2,tnvn]
            # positive3  (t_cls, v_g)  -> (t_cls, v_g')
            t2vg_sim = self._loose_similarity_row(sequence_output_batch, visual_global, attention_mask_batch,
                                                  video_mask_batch, sim_header=self.sim_header)
            t2vg_sim = t2vg_sim.reshape(bt, bv)
            # partial sim3a  (t_cls,v_g')
            t2vgh_sim = self._loose_similarity_row(sequence_output_batch, visual_global_partial, attention_mask_batch,
                                                   video_mask_batch, sim_header=self.sim_header)
            t2vgh_sim = t2vgh_sim.reshape(bv, bt).T  # [t1v1, t2v1, tnv1,...,t1vn,t2vn,tnvn]
            # partial sim2a  (t_g,v_tokens')
            tg2vhh_sim = self._loose_similarity_row(sequence_global, visual_token_hidden_partial, attention_mask_row,
                                                    video_mask_row, sim_header=self.sim_header)
            tg2vhh_sim = tg2vhh_sim.reshape(bv, bt).T  # [t1v1, t2v1, tnv1,...,t1vn,t2vn,tnvn]
            # partial sim2b  (t_g',v_tokens)
            tgh2vh_sim = self._loose_similarity_row(sequence_global_partial, visual_output_batch, attention_mask_row,
                                                    video_mask_batch, sim_header=self.sim_header)
            tgh2vh_sim = tgh2vh_sim.reshape(bt, bv)  # [t1v1,t1v2,t1vn,...,tnv1,tnv2,tnvn]
        return t2vh_sim, t2vhh_sim, tg2vh_sim, tg2vhh_sim, tgh2vh_sim, t2vg_sim, t2vgh_sim

    def _loose_similarity_row(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        """
            implemented ["wti", "ti"] interactions
            Args:
                sequence_output: repeat by batch or row, only sequence_output[0] works
                visual_output: repeat by batch or row, only visual_output[1] works
                attention_mask: repeat according to sequence_output[0]
                video_mask: repeat according to visual_output[1]
        """
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if 'ti' in self.interaction:
            sim_matrix_semantic = self.wti_interaction_row(sequence_output, visual_output,
                                                           attention_mask, video_mask)
        else:
            raise NotImplementedError("interaction:{} not implemented".format(self.interaction))
        return sim_matrix_semantic

    def wti_interaction_row(self, text_feat, video_feat, text_mask, video_mask):
        #########################################################
        # video mask need to change due to more tokens in frame #
        #########################################################
        if video_mask.shape[1] != video_feat.shape[1] and video_mask.shape[1] > video_feat.shape[1]:
            video_mask = video_mask[:, 0].reshape(video_feat.shape[0], -1)
        else:
            expand_times = video_feat.shape[1] // video_mask.shape[1]
            # video_mask shape here is (bs, max_frames)
            video_mask = video_mask.unsqueeze(1).repeat(1, 1, expand_times).view(video_mask.shape[0], -1)
        #########################################################
        # video mask need to change due to more tokens in frame #
        #########################################################
        #########################################################
        # text mask need to merge due to less tokens in sentence #
        #########################################################
        if text_mask.shape[1] != text_feat.shape[1]:
            text_mask = text_mask[:, 0].reshape(text_feat.shape[0], -1)
        if "wti" in self.interaction:
            text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
            video_weight = self.video_weight_fc(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
            video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool), float("-inf"))
            video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
        retrieve_logits = torch.einsum('ctd,cvd->ctv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('ctv,ct->ctv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('ctv,cv->ctv', [retrieve_logits, video_mask])
        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # ctv -> ct
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # ctv -> cv
        # max for video token
        if "wti" in self.interaction:  # weighted token-wise interaction
            t2v_logits = torch.einsum('ct,bt->c', [t2v_logits, text_weight])
            v2t_logits = torch.einsum('cv,bv->c', [v2t_logits, video_weight])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        else:  # token-wise interaction
            t2v_logits = torch.sum(t2v_logits, dim=1) / text_sum
            v2t_logits = torch.sum(v2t_logits, dim=1) / video_sum
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        return retrieve_logits

"""
LOSS FUNCTION
"""
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()
    def forward(self, sim_matrix, logit_scale=100.0):
        logpt = F.log_softmax(sim_matrix * logit_scale, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        # here means add a square in sim matrix to enlarge the sim score
        sim_loss = nce_loss.mean()
        return sim_loss

class NegNCE(nn.Module):
    def __init__(self, ):
        super(NegNCE, self).__init__()
        self.c_pos_w = 1.0
        self.c_neg_w = 0.5
        self.margin = 0.0
    def forward(self, sim_matrix, logit_scale=100.0):  # temp = 100, refer to X-CLIP
        # sim_matrix: [batch_t, batch_v]
        logpt = F.softmax(sim_matrix * logit_scale, dim=-1)
        logpt = torch.clamp(logpt, 1e-6, 1 - 1e-6)
        positive_logit = torch.diag(logpt).unsqueeze(-1)
        mask = (torch.eye(logpt.size(0)) > .5).to(logpt.device)
        d1 = torch.diag(sim_matrix)
        x = sim_matrix
        max_margin = F.relu(self.margin + x - d1.view(-1, 1)) + F.relu(self.margin + x - d1.view(1, -1))
        max_margin = max_margin.masked_fill(mask, 0)
        hard_negative_logits = logpt[max_margin > 0.]
        # local_cross_entropy
        loss_pos = - torch.log(positive_logit)
        loss_neg = - torch.log(1 - hard_negative_logits)
        if len(loss_neg) > 0:
            sim_loss = self.c_pos_w * loss_pos.mean() + self.c_neg_w * loss_neg.mean()
        else:
            sim_loss = self.c_pos_w * loss_pos.mean()
        return sim_loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

"""
SeqTransf
"""
class LayerNormDmae(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNormDmae, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class ResidualAttentionBlockDmae(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNormDmae(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNormDmae(d_model)
        self.n_head = n_head
    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class TransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDmae(width, heads) for _ in range(layers)])
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]