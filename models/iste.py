import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models
from models import mlp
from models import register
from utils import make_coord
from models import pixattn

import numpy as np


@register('iste')
class ISTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, local_ensemble=True, feat_unfold=True, cell_decode=True,
                 hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        # self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        # self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim*2, 3, padding=1)
        # self.phase = nn.Linear(2, hidden_dim//2, bias=False)
        self.liifconv = nn.Conv2d(self.encoder.out_dim, 256, 3, padding=1)
        self.coef = nn.Conv2d(self.encoder.out_dim, 256, 3, padding=1)
        self.freq1 = nn.Conv2d(self.encoder.out_dim, 256, 3, padding=1)
        self.freq2 = nn.Conv2d(self.encoder.out_dim, 256, 3, padding=1)
        self.phase = nn.Linear(2, 256, bias=False)
        self.toone = nn.Sigmoid()
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.m = 0.999
        self.imnet_texture = models.make(imnet_spec, args={'in_dim': 256})
        self.imnet_texture_auxiliary = models.make(imnet_spec, args={'in_dim': 256})
        self.imnet_rgb = mlp.MLP(in_dim=260, out_dim=3, hidden_list=[256, 256, 256])
        self.imnet_query = mlp.MLP(in_dim=3, out_dim=256, hidden_list=[256, 256, 256])
        self.pix_attn = pixattn.pix_attn(is_pooling=True)
        self.fusionlinear = nn.Linear(512, 256, bias=False)
        #self.lte_query = nn.Linear(2, hidden_dim)
        #self.lte_value = nn.Linear(64, hidden_dim)
        #self.lte_key = nn.Linear(64, hidden_dim)

    def _momentum_update_rgb_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_texture, param_texture_auxiliary in zip(self.imnet_texture.parameters(),
                                                          self.imnet_texture_auxiliary.parameters()):
            param_texture.data = param_texture.data * self.m + param_texture_auxiliary.data * (1. - self.m)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq1 = self.freq1(self.feat)
        self.freqq2 = self.freq2(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.pix_attn(self.feat)

        if self.imnet_rgb is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret


        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])


        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet_rgb(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def query_texture(self, coord, cell=None):
        feat = self.feat
        # coef = self.coeff
        # freq = self.freqq
        coef = self.coeff
        freq1 = self.freqq1
        freq2 = self.freqq2

        if self.imnet_rgb is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        '''
        if self.feat_unfold:
            feat_liif = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        '''

        feat = self.pix_attn(self.feat)
        #feat_liif = self.feat

        feat_liif = self.liifconv(feat)

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # feat_cell_s = torch.tensor(4)
        feat_coord = self.feat_coord

        preds_texture = []
        preds_rgb = []
        areas = []

        # with torch.no_grad():  # no gradient to keys
        #    self._momentum_update_key_encoder()

        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat_liif_key = F.grid_sample(
                    feat_liif, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                '''
                lte_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                '''

                q_freq1 = F.grid_sample(
                    freq1, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq2 = F.grid_sample(
                    freq2, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                # rel_coord2 =  coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                # rel_coord_s = rel_coord[:, :, 0] * rel_coord[:, :, 1]
                bs, q = coord.shape[:2]
                # prepare cell

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]


                # basis generation

                '''
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                phase = 2 * np.pi * (rel_coord_s/feat_cell_s).unsqueeze(-1)
                #* torch.ones(bs,q).cuda()
                q_freq = torch.sin(np.pi*q_freq + phase)
                '''

                # q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)

                # q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)
                '''
                lte_query = self.lte_query(rel_coord)
                lte_value = self.lte_value(lte_feat)
                lte_key = self.lte_key(lte_feat)
                lte_phase = self.phase(rel_cell)

                inp_texture = lte_value * torch.sin(lte_query * lte_key + lte_phase)
                '''

                q_freq = torch.stack((q_freq1, q_freq2), dim=-2)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                phase = self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                phase = 2 * np.pi * self.toone(phase)
                # * torch.ones(bs,q).cuda()
                q_freq = torch.sin(q_freq + phase)

                inp_texture = torch.mul(q_coef, q_freq)

                # K_unfold = K_unfold.permute(0, 2, 1)


                K = inp_texture
                Q = q_feat_liif_key.permute(0, 2, 1)
                V = inp_texture.permute(0, 2, 1)

                KF = F.normalize(K, dim=2)  # [N, Hr*Wr, C*k*k]
                QF = F.normalize(Q, dim=1)  # [N, C*k*k, H*W]

                R_lv3 = torch.bmm(KF, QF)  # [N, Hr*Wr, H*W]
                R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

                T_lv3 = self.bis(V, 2, R_lv3_star_arg)

                T = torch.cat([Q, T_lv3], 1)
                T = T.permute(0, 2, 1)
                T = self.fusionlinear(T)
                inp_feat = q_feat_liif_key + T * R_lv3_star.unsqueeze(-1)

                # pred_inp = inp.contiguous().view(bs * q, -1)

                inp_liif = torch.cat([inp_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp_liif = torch.cat([inp_liif, rel_cell], dim=-1)



                pred_rgb = self.imnet_rgb(inp_liif.view(bs * q, -1)).view(bs, q, -1)
                preds_rgb.append(pred_rgb)



                '''
                attention_scores = torch.sum(Query * Key, dim=-1) / math.sqrt(Key.shape[-1])
                attention_scores = nn.Softmax(dim=-1)(attention_scores)
                inp_texture = attention_scores.unsqueeze(-1) * Value
                '''

                pred_texture = self.imnet_texture(inp_texture.contiguous().view(bs * q, -1)).view(bs, q, -1)

                preds_texture.append(pred_texture)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];  areas[0] = areas[3];  areas[3] = t
        t = areas[1];  areas[1] = areas[2];  areas[2] = t

        ret_texture = 0
        ret_rgb = 0

        for pred_texture, area in zip(preds_texture, areas):
            ret_texture = ret_texture + pred_texture * (area / tot_area).unsqueeze(-1)

        for pred_rgb, area in zip(preds_rgb, areas):
            ret_rgb = ret_rgb + pred_rgb * (area / tot_area).unsqueeze(-1)

        output = ret_texture + ret_rgb
        '''
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        '''
        return output

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        output = self.query_texture(coord, cell)
        return output
