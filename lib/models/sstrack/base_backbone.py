from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.sstrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None
        self.template_lvl_embed = None
        self.search_lvl_embed = None
        self.use_lvl_embed = True

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_sep_seg = False

    def _apply_template_position_embed(self, z, B, T_z):
        """Single-frame spatial pos + per-template-frame lvl on (B*T_z, L, C)."""
        L = z.shape[1]
        if self.use_lvl_embed:
            z = z.view(B, T_z, L, -1)
            z = z + self.pos_embed_z.unsqueeze(1)
            if self.template_lvl_embed is not None:
                z = z + self.template_lvl_embed[:, :T_z, :].unsqueeze(2)
            return z.flatten(1, 2)
        z = z + self.pos_embed_z
        if T_z > 1:
            z = z.view(B, T_z, L, -1).flatten(1, 2)
        return z

    def _apply_search_position_embed(self, x, num_searches):
        """Single-frame spatial pos + per-search-frame lvl on concatenated search tokens."""
        B, _, C = x.shape
        L = x.shape[1] // num_searches
        if self.use_lvl_embed:
            x = x.view(B, num_searches, L, -1)
            x = x + self.pos_embed_x.unsqueeze(1)
            if self.search_lvl_embed is not None:
                x = x + self.search_lvl_embed[:, :num_searches, :].unsqueeze(2)
            return x.flatten(1, 2)
        return x + self.pos_embed_x

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG
        self.use_lvl_embed = bool(getattr(cfg.MODEL.BACKBONE, 'USE_LVL_EMBED', True))
        num_template_frames = int(getattr(cfg.DATA.TEMPLATE, 'NUMBER', 1))
        num_search_frames = int(cfg.DATA.SEARCH.LENGTH)

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region (single-frame spatial grid; frame index via search_lvl_embed)
        H, W = search_size
        if self.use_lvl_embed:
            new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        else:
            new_P_H, new_P_W = num_search_frames * H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        if self.use_lvl_embed:
            self.template_lvl_embed = nn.Parameter(
                torch.zeros(1, num_template_frames, self.embed_dim))
            self.search_lvl_embed = nn.Parameter(
                torch.zeros(1, num_search_frames, self.embed_dim))
            trunc_normal_(self.template_lvl_embed, std=.02)
            trunc_normal_(self.search_lvl_embed, std=.02)
        else:
            self.template_lvl_embed = None
            self.search_lvl_embed = None

        # for cls token and track query token positional embeddings
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)
            self.track_pos_embed = nn.Parameter(torch.zeros_like(cls_pos_embed))
            trunc_normal_(self.track_pos_embed, std=.02)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def _build_prefix_tokens(
            self, B, track_query=None, token_type="add", token_len=1,
            separate_track_cls=False, track_query_len=1, cls_token_len=1,
    ):
        """Build prefix tokens: [track..., cls...] when separated, else legacy merged query."""
        if not self.add_cls_token:
            return None, 0, 0, 0

        if separate_track_cls:
            init_track = self.track_query_token.expand(B, track_query_len, -1)
            if track_query is not None:
                prev_len = track_query.size(1)
                if token_type == "concat":
                    if prev_len >= track_query_len:
                        track_tokens = track_query[:, :track_query_len, :]
                    else:
                        pad = init_track[:, :track_query_len - prev_len, :]
                        track_tokens = torch.cat([pad, track_query], dim=1)
                elif token_type == "add":
                    if prev_len == track_query_len:
                        track_tokens = init_track + track_query
                    elif prev_len < track_query_len:
                        pad = init_track[:, :track_query_len - prev_len, :]
                        track_tokens = torch.cat([pad, track_query], dim=1) + init_track
                    else:
                        track_tokens = init_track + track_query[:, :track_query_len, :]
                else:
                    raise ValueError(f"Unknown token_type: {token_type!r}, expected 'concat' or 'add'")
            else:
                track_tokens = init_track
            track_tokens = track_tokens + self.track_pos_embed

            cls_tokens = self.cls_token.expand(B, cls_token_len, -1) + self.cls_pos_embed
            query = torch.cat([track_tokens, cls_tokens], dim=1)
            query_len = track_query_len + cls_token_len
            return query, query_len, track_query_len, cls_token_len

        if token_type == "concat":
            if track_query is None:
                query = self.cls_token.expand(B, token_len, -1)
            else:
                track_len = track_query.size(1)
                new_query = self.cls_token.expand(B, token_len - track_len, -1)
                query = torch.cat([new_query, track_query], dim=1)
        elif token_type == "add":
            new_query = self.cls_token.expand(B, token_len, -1)
            query = new_query if track_query is None else track_query + new_query
        else:
            raise ValueError(f"Unknown token_type: {token_type!r}, expected 'concat' or 'add'")
        query = query + self.cls_pos_embed
        return query, token_len, 0, token_len

    def _block_kwargs(self, add_cls_token, query_len, lens_z, lens_x,
                      separate_track_cls, track_query_len, cls_token_len):
        if not add_cls_token:
            return {"add_cls_token": False}
        return {
            "add_cls_token": True,
            "query_len": query_len,
            "lens_z": lens_z,
            "lens_x": lens_x,
            "separate_track_cls": separate_track_cls,
            "track_query_len": track_query_len,
            "cls_token_len": cls_token_len,
        }

    def forward_features(self, z, xs, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1,
                         separate_track_cls=False, track_query_len=1, cls_token_len=1,
                         ):
        B, H, W = xs[-1].shape[0], xs[-1].shape[2], xs[-1].shape[3]
        num_searches = len(xs)
        
        x = self.patch_embed(xs[-1])
        for ind in range(num_searches-1, 0, -1):
            x_ = self.patch_embed(xs[ind-1])
            x = torch.cat((x_,x), dim=1)
        top_k_indices = None
        
        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            query, query_len, eff_track_len, eff_cls_len = self._build_prefix_tokens(
                B, track_query, token_type, token_len,
                separate_track_cls, track_query_len, cls_token_len,
            )
        else:
            query, query_len, eff_track_len, eff_cls_len = None, 0, 0, 0
        
        z = self._apply_template_position_embed(z, B, T_z)
        x = self._apply_search_position_embed(x, num_searches)

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        lens_z = z.shape[1]  # HW
        lens_x = x.shape[1]  # HW

        x = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)     # (B, prefix+z+x, 768)
        x = self.pos_drop(x)

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        blk_kw = self._block_kwargs(
            self.add_cls_token, query_len, lens_z, lens_x,
            separate_track_cls, eff_track_len, eff_cls_len,
        )
        
        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, **blk_kw)
                
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        if self.add_cls_token:
            query = x[:, :query_len]
            z = x[:, query_len:lens_z_new+query_len]
            x = x[:, lens_z_new+query_len:]
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([query, z, x], dim=1)

        # aux_dict = {}
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict, top_k_indices

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None,
                token_type="add", token_len=1,
                separate_track_cls=False, track_query_len=1, cls_token_len=1):
        x, aux_dict, top_k_indices = self.forward_features(
            z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
            track_query=track_query, token_type=token_type, token_len=token_len,
            separate_track_cls=separate_track_cls, track_query_len=track_query_len,
            cls_token_len=cls_token_len,
        )
        return x, aux_dict, top_k_indices
