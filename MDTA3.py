import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
from einops import rearrange
from module.model_libs import CBAM,Conv3x3,RFB_modified,RCAB
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.hint_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


    def forward(self, x,y):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        # hint=self.hint_dwconv(y)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # hint = rearrange(hint, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # hint=torch.nn.functional.normalize(hint, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        
        return out
    
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)    

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.hint=nn.Conv2d(1,dim,1)
        self.norm3=LayerNorm(dim,LayerNorm_type)
    
    def forward(self, x,y):
        
        b,c,h,w=x.size()
        y=F.interpolate(y,size=(h,w))
        y=self.hint(y)
        x = x + self.attn(self.norm1(x),self.norm3(y))
        x = x + self.ffn(self.norm2(x))

        return x
class CustomSequential(nn.Sequential):
    def forward(self, x, y):
        for module in self:
            x = module(x, y)
        return x
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.CHlayer=RCAB(dim)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        
        x = self.project_out(x)
        x = self.CHlayer(x)
        
        return x
class Upsample2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample2, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, C,H,W = x.shape
        out = self.deconv(x)# B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DinatBlock(nn.Module):
    # SAMBA_SUCK models/encoders/dinat/dinats.py NATransformerLayer, B C H W in/out
    def __init__(self, dim, num_heads, kernel_size=7, dilation=1, mlp_ratio=2.66, qkv_bias=True, drop_path=0.0, norm_layer=nn.LayerNorm):
        super(DinatBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(dim, kernel_size=kernel_size, dilation=dilation, num_heads=num_heads,
                                          qkv_bias=qkv_bias, rel_pos_bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, y):
        b,c,h,w = x.size()
        # 특징맵이 kernel*dilation 보다 작으면 dilation을 줄여서 사용
        self.attn.dilation = max(1, min(self.dilation, min(h, w) // self.kernel_size))
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2).contiguous()
def attn_dinat_blocks(dim, num_heads, num_blocks, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', kernel_size=7, dilation=1):
    # attention_block(MDTA) -> dinat_block 순서로 번갈아 쌓음, dinat은 dilation 1 / dilation 번갈아 사용 (DiNAT 방식)
    blocks = []
    for i in range(num_blocks):
        if i % 2 == 0:
            blocks.append(TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        else:
            blocks.append(DinatBlock(dim=dim, num_heads=num_heads, kernel_size=kernel_size, dilation=1 if (i // 2) % 2 == 0 else dilation, mlp_ratio=ffn_expansion_factor))
    return CustomSequential(*blocks)
class Restormer_Decoder(nn.Module):
    def __init__(self, 
        dim = 256,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'   ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        kernel_size = 7,
        dilations = [1,2,4,4],  ## latent, level1, level2, level3/refinement
    ):
        super(Restormer_Decoder, self).__init__()
        
        self.up1= Upsample(int(dim))
        self.up2= Upsample(int(dim))
        self.up3= Upsample(int(dim))
        self.up4= Upsample(int(dim))
        
        
        
        
        self.reduce_chan_level1 = nn.Conv2d(int(dim/2*3), int(dim), kernel_size=1, bias=bias)
        
        self.reduce_chan_level2 = nn.Conv2d(int(dim/2*3), int(dim), kernel_size=1, bias=bias)
        
        self.reduce_chan_level3 = nn.Conv2d(int(dim/2*3), int(dim), kernel_size=1, bias=bias)
        
        self.latent = attn_dinat_blocks(int(dim), heads[3], num_blocks[3], ffn_expansion_factor, bias, LayerNorm_type, kernel_size, dilations[0])
        
        self.decoder_level1 = attn_dinat_blocks(int(dim), heads[2], num_blocks[2], ffn_expansion_factor, bias, LayerNorm_type, kernel_size, dilations[1])
        
        self.decoder_level2 = attn_dinat_blocks(int(dim), heads[1], num_blocks[1], ffn_expansion_factor, bias, LayerNorm_type, kernel_size, dilations[2])
        
        self.decoder_level3 = attn_dinat_blocks(int(dim), heads[0], num_blocks[0], ffn_expansion_factor, bias, LayerNorm_type, kernel_size, dilations[3])
        
        # self.decoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = attn_dinat_blocks(int(dim), heads[0], num_refinement_blocks, ffn_expansion_factor, bias, LayerNorm_type, kernel_size, dilations[3])
    
    def forward(self,list,y):
        x1,x2,x3,x4=list
        
        H,W=x1.shape[2], x1.shape[3]
        self.image_size=(H,W)
        
        x1_1=self.latent(x1,y)
        
        
        x1_1=self.up1(x1_1)
        
        x2_1=torch.cat([x1_1,x2],dim=1)
        
        
        x2_1=self.reduce_chan_level1(x2_1)
        x2_2=self.decoder_level1(x2_1,y)
        
        
        x2_2=self.up2(x2_2)
        
        x3_1=torch.cat([x2_2,x3],dim=1)
        x3_1=self.reduce_chan_level2(x3_1)
        x3_2=self.decoder_level2(x3_1,y)
        
        x3_2=self.up3(x3_2)
        # x3_2=self.up2(x3_2)
        x4_1=torch.cat([x3_2,x4],dim=1)
        x4_1=self.reduce_chan_level3(x4_1)
        x4_2=self.decoder_level3(x4_1,y)
        
        refine=self.refinement(x4_2,y)
        
        return refine
