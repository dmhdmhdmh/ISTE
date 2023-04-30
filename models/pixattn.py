import numbers
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import as_strided
from einops import rearrange
from models import register

class PatchEmbeding(nn.Module):
    r"""Image to Patch Embedding

    (B,C,H,W) -> (B,H/patch_size * W/patch_size, embed_dim)

    Args:
        img_size: Image size. Default 224
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_channs=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size)
        patch_size = (patch_size)
        patch_resolution = [img_size // patch_size, img_size // patch_size]
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution * patch_resolution

        self.proj = nn.Conv2d(in_channs, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input Image Size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x).view(B, self.embed_dim, -1).transpose(1, 2)  # (B,C,H,W) -> (B,H/ps * W/ps,embed_dim)

        if self.norm is not None:
            x = self.norm(x)

        return x

@register('pixattn')
class pix_attn(nn.Module):

    def __init__(self, in_dim=64, num_heads=8, input_resolution=64, qkv_scale=None, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., g_pooling=nn.AdaptiveAvgPool1d(1), pooling_size=1, window_size=3,
                 padding_turple=(1, 1, 1, 1), is_pooling=True):
        super().__init__()

        self.is_pooling = is_pooling
        self.window_size = window_size
        self.in_dim = in_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.qk_linear = nn.Linear(in_dim, 2 * in_dim, bias=qkv_bias)
        self.g_pooling = g_pooling
        self.pooling = nn.AvgPool2d((pooling_size, pooling_size), (pooling_size, pooling_size))
        self.kv_linear = nn.Linear(in_dim, 2 * in_dim, bias=qkv_bias)
        self.scale = qkv_scale or (in_dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        ## depth-wise conv
        self.depth_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=pooling_size,
                                    stride=pooling_size,
                                    padding=0,
                                    groups=in_dim)

        self.reflec_pad = nn.ReflectionPad2d(padding=padding_turple)

    def view_as_windows(self, arr_in, window_size, step=1):

        # -- basic checks on arguments
        if not isinstance(arr_in, np.ndarray):
            raise TypeError("`arr_in` must be a numpy ndarray")

        ndim = arr_in.ndim

        if isinstance(window_size, numbers.Number):
            window_size = (window_size,) * ndim
        if not (len(window_size) == ndim):
            raise ValueError("`window_size` is incompatible with `arr_in.shape`")

        if isinstance(step, numbers.Number):
            if step < 1:
                raise ValueError("`step` must be >= 1")
            step = (step,) * ndim
        if len(step) != ndim:
            raise ValueError("`step` is incompatible with `arr_in.shape`")

        arr_shape = np.array(arr_in.shape)
        window_size = np.array(window_size, dtype=arr_shape.dtype)

        if ((arr_shape - window_size) < 0).any():
            raise ValueError("`window_size` is too large")

        if ((window_size - 1) < 0).any():
            raise ValueError("`window_size` is too small")

        # -- build rolling window view
        slices = tuple(slice(None, None, st) for st in step)
        window_strides = np.array(arr_in.strides)

        indexing_strides = arr_in[slices].strides

        win_indices_shape = (
                                    (np.array(arr_in.shape) - np.array(window_size)) // np.array(step)
                            ) + 1

        #win_indices_shape = np.array([4, 48, 48])
        new_shape = tuple(list(win_indices_shape) + list(window_size))
        strides = tuple(list(indexing_strides) + list(window_strides))

        arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
        return arr_out

    def enpadding_process(self, x):
        x = self.reflec_pad(x)  # padding (48,48) -> (63,63)
        B, C, H, W = x.size()
        x = x.reshape(B * C, H, W).cpu().detach().numpy()
        #x = x.reshape(B * C, H, W).numpy()
        x = self.view_as_windows(x, (B * C, self.window_size, self.window_size), step=1)
        #x = rearrange(x,"b n1 n2 c h w -> b (n1 n2) c h w") #(b,48*48,64,16,16)
        x = rearrange(x, "1 n1 n2 (b c) h w -> b (n1 n2) c h w", b=B, c=C)  # (b,48*48,64,16,16)
        x = torch.tensor(x).cuda()
        #x = torch.tensor(x)
        #x = torch.split(x, C, dim=2)
        #x = torch.cat(x, dim=0)
        return x

    def forward(self, x):
        '''
        padding and spliting window
        input: raw image with size (1,64,48,48)
        output: size (1,64,48,48)


        x_raw: each token in a whole image
        x    : each window split from the whole image
        b    : batch size
        B    : the number of windows is b * image_resolution * image_resolution

        '''
        B_, _, H_, W_ = x.size()
        x_raw = rearrange(x, "b c h w -> (b h w) 1 c ")  # (b*48*48,1,64)***********************
        x = self.enpadding_process(x)
        x = rearrange(x, "b n c h w -> (b n) c h w")  # (b*48*48,64,16,16)
        B, C, H, W = x.size()
        L = H * W
        x_ = x.reshape(B, C, self.window_size, self.window_size)  # (B,C,16,16)
        x = x.reshape(B, L, C)

        # K & V
        if self.is_pooling == True:
            x_f = self.pooling(x_)  # (B,C,4,4)
            x_f = x_f.view(B, -1, C)  # (b*48*48,4*4,64) ************************
            # insert global info
            if self.g_pooling:
                x_g = self.g_pooling(x.transpose(1, 2)).transpose(1, 2)  # (B,1,C)
                x_f = torch.cat((x_g, x_f), dim=1)  # (B,17,C)
        else:  # depth-conv
            x_f = self.depth_conv(x_)
            x_f = x_f.view(B, -1, C)  # (B,8*8,C)

        q_a = self.qk_linear(x_raw).reshape(B, 1, 2, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        q, q_ = q_a[0], q_a[1]
        q_insert = q * q_  # (B,n,L,d)
        q_value = q_insert.sum(-1).unsqueeze(-1)  # (B,n,L,1)

        kv = self.kv_linear(x_f).reshape(B, x_f.size(1), 2, self.num_heads, self.in_dim // self.num_heads).permute(2, 0,
                                                                                                                   3, 1,
                                                                                                                   4)
        k, v = kv[0], kv[1]  # (B,n,l,d)

        # attn
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B,n,L,l)
        # insert self info
        attn = torch.cat((attn, q_value), dim=-1)  # (B,n,L,l+1)

        attn = self.softmax(attn)

        # value
        x_fusion = (attn[:, :, :, :-1] @ v).transpose(1, 2).reshape(B, 1, C)
        x_sole = (attn[:, :, :, -1].unsqueeze(-1) * q_).transpose(1, 2).reshape(B, 1, C)
        x = x_sole + x_fusion
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "(b h w) 1 c -> b (1 c) h w", h=H_, w=W_, b=B_)

        return x


if __name__ == "__main__":
    input = torch.rand(4, 64, 48, 48)
    pix_attn = pix_attn(is_pooling=True)
    y = pix_attn(input)
    print(y.size())