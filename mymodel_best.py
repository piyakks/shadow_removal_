import os
import torch
from torch import nn
import torch.nn.functional as F
from module.model_libs import RFB_modified
from sam2_modeling.backbones.image_encoder import ImageEncoder,FpnNeck
from sam2_modeling.backbones.hieradet_adapter import Hiera

from sam2_modeling.position_encoding import PositionEmbeddingSine
try:
    from sam_modeling.image_encoder import ImageEncoderViT
except ImportError:
    ImageEncoderViT = None
from functools import partial
from module.TransformerDecoder import TransformerDecoder
import copy

from MDTA3 import Restormer_Decoder,Upsample,TransformerBlock,attn_dinat_blocks

def load_pretrain(mode,encoder):
    if mode=='sam_h':
        missing_keys, unexpected_keys=encoder.load_state_dict(torch.load('sam_weight/sam_h_encoder.pth'),strict=False)
    elif mode in ['hiera_adapter','heira_l_adapter']:
        # 변환된 heira_l_adapter.pth 또는 공식 sam2_hiera_large.pt 둘 다 사용 가능
        path='sam_weight/heira_l_adapter.pth'
        if not os.path.exists(path):
            path='sam_weight/sam2_hiera_large.pt'
        checkpoint=torch.load(path,map_location='cpu')
        if 'model' in checkpoint:
            checkpoint=checkpoint['model']
        prefix='image_encoder.trunk.'
        if any(k.startswith(prefix) for k in checkpoint):
            checkpoint={k[len(prefix):]:v for k,v in checkpoint.items() if k.startswith(prefix)}
        # patch_embed 3ch(RGB) -> 4ch(RGB+mask): RGB는 그대로, mask 채널은 0
        w=checkpoint.get('patch_embed.proj.weight')
        if w is not None and w.shape[1]==3:
            checkpoint['patch_embed.proj.weight']=torch.cat([w,torch.zeros_like(w[:,:1])],dim=1)
        missing_keys, unexpected_keys=encoder.load_state_dict(checkpoint,strict=False)
        print('missing keys: %d, unexpected keys: %d'%(len(missing_keys),len(unexpected_keys)))
    elif mode=='sam2_l_adapter':
        checkpoint=torch.load('sam_weight/sam2_l_encoder.pth')
        if 'trunk.patch_embed.proj.weight' in checkpoint:
            del checkpoint['trunk.patch_embed.proj.weight']
        missing_keys, unexpected_keys=encoder.load_state_dict(checkpoint,strict=False)
        
    elif mode=='heira':
        missing_keys, unexpected_keys=encoder.load_state_dict(torch.load('sam_weight/sam2_l_encoder.pth'),strict=False)
    else:
        print('failed')
        
    return missing_keys,encoder
def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
class CustomSequential(nn.Sequential):
    def forward(self, x, y):
        for module in self:
            x = module(x, y)
        return x
class mymodel(nn.Module):
    def __init__(self,mode='sam_h',pretrained=False):
        super(mymodel,self,).__init__()
        self.mode=mode
        
        # self.hiera = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")#hiera
        self.backbone_list=None
            
        if self.mode=='sam_h':
            self.encoder=ImageEncoderViT(depth=32,embed_dim=1280,img_size=1024,mlp_ratio=4,norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
                                ,num_heads=16,patch_size=16,qkv_bias=True,use_rel_pos=True,global_attn_indexes=[7,15,23,31],window_size=14,out_chans=256)
        elif self.mode in ['hiera_adapter','heira_l_adapter']:
            self.encoder=Hiera(in_chans=4,embed_dim=144,
                                    num_heads=2,
                                    stages=[2,6,36,4],
                                    global_att_blocks=[22,33,43],
                                    window_pos_embed_bkg_spatial_size=[7,7],
                                    window_spec=[8,4,16,8],img_size=512)

        elif self.mode=='sam2_l_adapter':
            self.backbone_list=[1152, 576, 288, 144]
            self.encoder=ImageEncoder(
                                    trunk=Hiera(in_chans=4,embed_dim=144,
                                    num_heads=2,
                                    stages=[2,6,36,4],
                                    global_att_blocks=[22,33,43],
                                    window_pos_embed_bkg_spatial_size=[7,7],
                                    window_spec=[8,4,16,8],img_size=512),
                                    neck=FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256,normalize=True,scale=None,temperature=1000),
                                    d_model=256,backbone_channel_list= self.backbone_list,
                                    fpn_top_down_levels=[2,3],fpn_interp_model='nearest'))
            
            
       
            
        if pretrained==True:
            missing_keys,self.encoder=load_pretrain(self.mode,self.encoder)
            for param in self.encoder.parameters():
                param.requires_grad = False
            for name, param in self.encoder.named_parameters():
                if any(missing_key in name for missing_key in missing_keys):
                    param.requires_grad = True
            if self.mode=='heira_l_adapter':
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = name.startswith('prompt_generator.') or name.startswith('patch_embed.')

        # hiera 4단계 feature [144,288,576,1152] -> 각 단계 decoder 채널에 맞춤 [32,64,128,256]
        self.feat_proj=nn.ModuleList([nn.Conv2d(c,d,1) for c,d in zip([144,288,576,1152],[32,64,128,256])])

        self.up1=Upsample(256)# 256 -> 128, H/32 -> H/16
        self.up2=Upsample(128)# 128 -> 64,  H/16 -> H/8
        self.up3=Upsample(64)# 64 -> 32,    H/8 -> H/4
        self.reduce1=nn.Conv2d(256,128,1)# concat(128+128) -> 128
        self.reduce2=nn.Conv2d(128,64,1)# concat(64+64) -> 64
        self.decode_first = attn_dinat_blocks(256, 1, 4, dilation=4)#H/32
        self.decode_half = attn_dinat_blocks(128, 1, 4, dilation=4)#H/16
        self.decode_main = attn_dinat_blocks(64, 1, 4, dilation=4)#H/8
        self.decode_last = attn_dinat_blocks(64, 1, 4, dilation=4)#H/4, concat(32+32) 그대로 사용
        # H/4 -> H
        self.up4=Upsample(64)# 64 -> 32, H/2
        self.up5=Upsample(32)# 32 -> 16, H
        # self.cbam=CBAM()
        self.rgb_2_1=nn.Sequential(
            nn.Conv2d(16,3,3,1,1),
        )
        self.rgb_sig=nn.Sequential(
            nn.Conv2d(64,3,3,1,1),
        )
        self.sigmoid=nn.Sigmoid()
     
    def forward(self,x,y):
        H, W = x.shape[2], x.shape[3]
        xy=torch.cat([x,y],dim=1)
        
        t=self.encoder(xy)
        
        self.features = []
        
        
        #b_plus 모델은 4개, large 모델은 3개 #차원은 256 고정  이미지 사이즈 256~32 까지     
        for i in t:
            self.features.append(i)
        
        # 각 feature 채널을 맞추고 깊은 단계부터 unet처럼 올라감
        p1,p2,p3,p4=[proj(f) for proj,f in zip(self.feat_proj,self.features)]

        complex_map=self.decode_first(p4,y)                          # H/32, 256

        complex_map_shadow=self.up1(complex_map)
        complex_map_cat=self.reduce1(torch.cat([complex_map_shadow,p3],dim=1))

        complex_map_shadow=self.decode_half(complex_map_cat,y)       # H/16, 128

        complex_map=self.up2(complex_map_shadow)

        complex_map_cat=self.reduce2(torch.cat([complex_map,p2],dim=1))

        complex_map=self.decode_main(complex_map_cat,y)              # H/8, 64

        complex_map=self.up3(complex_map)
        complex_map_cat=torch.cat([complex_map,p1],dim=1)

        complex_map=self.decode_last(complex_map_cat,y)              # H/4, 64

        complex_map=self.up5(self.up4(complex_map))                  # H, 16

        complex_map_clamp=self.rgb_2_1(complex_map)+x
        
        # complex_map_sigmoid=torch.sigmoid(self.rgb_sig(complex_map)+x)

        if self.mode=='sam_h':
            return 0
           # 이러면 4계층으로 나눠주는것도 나쁘지않을듯 
        
        
        return complex_map_clamp#,complex_map_sigmoid
