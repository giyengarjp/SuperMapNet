import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.cross_encoder.Cross_Attn import LidarPred


class CrossEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEncoder, self).__init__()
        self.lidar_pred = LidarPred(**kwargs)


    def forward(self, inputs):
        '''img_feat = inputs["im_bkb_features"] [-1]
        img_feat = img_feat.reshape(*inputs['images'].shape[:2], *img_feat.shape[-3:])
        img_feat = img_feat.permute(0, 2, 3, 1, 4) 
        #print('3', img_feats.size())
        img_feat = img_feat.reshape(*img_feat.shape[:3],  -1)'''
        
        # Stack list of tensors back into batch: list([B x [C,H,W]]) -> [B,C,H,W]
        img_feat = inputs["img_enc_features"][-1]
        lidar_enc_feat_list = inputs["lidar_enc_features"]
        lidar_enc_feat = torch.stack(lidar_enc_feat_list, dim=0) if isinstance(lidar_enc_feat_list, list) else lidar_enc_feat_list[-1]
       
        img_bev_feat, lidar_feature = self.lidar_pred(lidar_enc_feat, img_feat)
        #print(lidar_feature.size())
        
        return {
            "img_bev_feat": img_bev_feat,
            "lidar_bev_feat": lidar_feature
        }

  