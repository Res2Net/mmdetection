import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None,
                 scale=4,
                 baseWidth=26,
                 stype='normal'):
        """Bottle2neck block for Res2Net.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottle2neck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, width * scale, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            width * scale,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []

        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            for i in range(self.nums):
                convs.append(
                    build_conv_layer(
                        conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=dilation,
                        dilation=dilation,
                        bias=False))
                bns.append(build_norm_layer(norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        else:
            assert self.conv_cfg is None, 'conv_cfg cannot be None for DCN'
            for i in range(self.nums):
                convs.append(
                    build_conv_layer(
                        dcn,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=dilation,
                        dilation=dilation,
                        bias=False))
                bns.append(build_norm_layer(norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

        self.conv3 = build_conv_layer(
            conv_cfg,
            width * scale,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            # out = self.conv2(out)
            # out = self.norm2(out)
            # out = self.relu(out)

            spx = torch.split(out, self.width, 1)
            for i in range(self.nums):
                if i == 0 or self.stype == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)
            if self.scale != 1 and self.stype == 'normal':
                out = torch.cat((out, spx[self.nums]), 1)
            elif self.scale != 1 and self.stype == 'stage':
                out = torch.cat((out, self.pool(spx[self.nums])), 1)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[],
                   scale=4,
                   baseWidth=26):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=True,
                count_include_pad=False),
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=1,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None,
            scale=scale,
            baseWidth=baseWidth,
            stype='stage'))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None,
                scale=scale,
                baseWidth=baseWidth))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class Res2Net(nn.Module):
    """Res2Net backbone.

    Args:
        depth (int): Depth of Res2Net, from {18, 34, 50, 101, 152}.
        scale (int): Scales used in Res2Net, normally 4.
        baseWidth (int): basic width of each scale, normally 26.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Res2Net stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import Res2Net
        >>> import torch
        >>> self = Res2Net(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (Bottle2neck, (3, 4, 6, 3)),
        101: (Bottle2neck, (3, 4, 23, 3)),
        152: (Bottle2neck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 scale=4,
                 baseWidth=26,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(Res2Net, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for res2net'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.scale = scale
        self.baseWidth = baseWidth
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer(in_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i],
                scale=scale,
                baseWidth=baseWidth)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels):
        self.conv1 = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
        )
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in self.conv1:
                for param in m.parameters():
                    param.requires_grad = False
            for param in self.norm1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            checkpoint = model_zoo.load_url(
                pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottle2neck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottle2neck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(Res2Net, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
