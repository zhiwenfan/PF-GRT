import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone import build_backbone

from ..transformer_network import (
    Transformer2D,
    Transformer,
    Embedder
)

from loguru import logger
import math

class NoPEGNT(nn.Module):
    def __init__(self, config, posenc_dim=3 + 3 * 2 * 10, viewenc_dim=3 + 3 * 2 * 10, ret_alpha=False):
        super(NoPEGNT, self).__init__()
        self.config = config
        self.netwidth = config["netwidth"]

        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(256, self.netwidth),
            nn.ReLU(),
            nn.Linear(self.netwidth, self.netwidth),
        )

        self.backbone = build_backbone(config)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        
        self.embedding_stdev = (1./math.sqrt(256))
        self.pixel_embedding = None
        self.canonical_camera_embedding = nn.Parameter(torch.randn(1, 1, 256) * self.embedding_stdev)
        self.non_canonical_camera_embedding = nn.Parameter(torch.randn(1, 1, 256) * self.embedding_stdev)

        self.norm = nn.LayerNorm(config["netwidth"])
        self.rgb_fc = nn.Linear(config["netwidth"], 3)

        # view self-atten and ray cross-atten
        self.view_selftrans = nn.ModuleList([])
        self.ray_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        for i in range(config["trans_depth"]):

            view_trans = Transformer(
                dim=config["netwidth"],
                ff_hid_dim=int(config["netwidth"] * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )

            self.ray_crosstrans.append(view_trans)

            # ray transformer
            ray_trans = Transformer(
                dim=config["netwidth"],
                ff_hid_dim=int(config["netwidth"] * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)
            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(config["netwidth"] + posenc_dim + viewenc_dim, config["netwidth"]),
                    nn.ReLU(),
                    nn.Linear(config["netwidth"], config["netwidth"]),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)


    def forward(self, src_images, pts, ray_d, dists=None, z_vals=None):
        """
        Args:
            src_images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """
        batch_size, num_images = src_images.shape[:2]
        x = src_images.flatten(0, 1)

        canonical_idxs = torch.zeros(batch_size, num_images)
        canonical_idxs[:, 0] = 1
        canonical_idxs = canonical_idxs.flatten(0, 1).unsqueeze(-1).unsqueeze(-1).to(x)
        camera_id_embedding = canonical_idxs * self.canonical_camera_embedding + \
                              (1. - canonical_idxs) * self.non_canonical_camera_embedding
        # 1/8      1/2
        feats_c, feats_f = self.backbone(x)

        if self.pixel_embedding is None:
            self.pixel_embedding = nn.Parameter(torch.randn(feats_c.shape) * self.embedding_stdev)
        
        feats_c = feats_c + self.pixel_embedding
        feats_c = feats_c.flatten(2, 3).permute(0, 2, 1)
        feats_c = feats_c + camera_id_embedding #[N, (W/8)x(H/8),256]

        depth_pred = None
        weight = None

        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)
        
        logger.info(f"shape: of feats_c: {feats_c.shape}, \
                                input_pts: {input_pts.shape}\
                                input_views:{input_views.shape}") 

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(feats_c) #[N,W/8*H/8, netwidth]

        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]
        rgb_feat = rgb_feat.flatten(0,1) #[xxx,netwidth]

        # transformer modules

        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.ray_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            q = crosstrans(q, rgb_feat)

            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)
                q = q_fc(q)
            # ray transformer
            q = selftrans(q, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q

        # normalize & rgb
        h = self.norm(q).         #[xxx,netwidth]
        last = h.mean(dim=1)
        rgb_pred = self.rgb_fc(last)
        logger.info(f"rgb_pred:{rgb_pred.shape}")
        if self.ret_alpha:
            return torch.cat([rgb_pred, attn], dim=1)
        else:
            return rgb_pred

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


