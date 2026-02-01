import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Re-parameterizable Conv Block
class RepConv(nn.Module):
    """
    Re-parameterizable Convolution Block
    훈련: Conv + BN + Identity(or 1x1) branch
    추론: 단일 3x3 Conv로 융합
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, use_identity=True, use_activation=True):
        super(RepConv, self).__init__()
       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
       
        # use_identity는 stride=1이고 in/out 채널이 같을 때만 활성화
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)
       
        # 주 Branch: kernel_size Conv + BN
        self.conv_kxk = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups, bias=False)
        self.bn_kxk = nn.BatchNorm2d(out_channels)
       
        # 1x1 Branch (더 많은 표현력)
        if kernel_size > 1:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1,
                                      stride, 0, groups=groups, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1 = None
       
        # Identity Branch (residual connection)
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
       
        # 활성화 함수 선택적 적용
        if use_activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()
           
    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            # 추론 모드: 융합된 단일 Conv만 사용
            return self.activation(self.fused_conv(x))
       
        # 훈련 모드: 모든 branch 합산
        out = self.bn_kxk(self.conv_kxk(x))
       
        if self.conv_1x1 is not None:
            out += self.bn_1x1(self.conv_1x1(x))
       
        if self.use_identity:
            out += self.bn_identity(x)
       
        return self.activation(out)
   
    def switch_to_deploy(self):
        """추론 모드로 전환: 모든 branch를 단일 Conv로 융합"""
        if hasattr(self, 'fused_conv'):
            return
       
        # 각 branch의 weight와 bias를 추출하여 합산
        kernel, bias = self._fuse_bn_tensor(self.conv_kxk, self.bn_kxk)
       
        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1, self.bn_1x1)
            # 1x1을 kxk로 패딩
            kernel += self._pad_1x1_to_kxk(kernel_1x1)
            bias += bias_1x1
       
        if self.use_identity:
            kernel_identity, bias_identity = self._fuse_bn_tensor(None, self.bn_identity)
            kernel += kernel_identity
            bias += bias_identity
       
        # 융합된 Conv 생성
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
       
        # 훈련용 레이어 제거 (메모리 절약)
        self.__delattr__('conv_kxk')
        self.__delattr__('bn_kxk')
        if self.conv_1x1 is not None:
            self.__delattr__('conv_1x1')
            self.__delattr__('bn_1x1')
        if hasattr(self, 'bn_identity'):
            self.__delattr__('bn_identity')
   
    def _fuse_bn_tensor(self, conv, bn):
        """Conv + BN을 융합하여 weight, bias 반환"""
        if conv is None:
            # Identity branch
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros((self.in_channels, input_dim,
                                        self.kernel_size, self.kernel_size),
                                       dtype=bn.weight.dtype, device=bn.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim,
                             self.kernel_size // 2, self.kernel_size // 2] = 1
            kernel = kernel_value
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        else:
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
       
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
   
    def _pad_1x1_to_kxk(self, kernel_1x1):
        """1x1 kernel을 kxk로 패딩"""
        if self.kernel_size == 1:
            return kernel_1x1
        else:
            pad = self.kernel_size // 2
            return F.pad(kernel_1x1, [pad, pad, pad, pad])

 
class GatedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=8, dropout_rate=0.2):
        super(GatedSpatialAttention, self).__init__()
        intermediate_channels = in_channels // reduction

        self.conv1_rep = RepConv(in_channels, intermediate_channels, 7, 1, 3, 
                                use_identity=True, use_activation=True)
        
        self.dropout_1 = nn.Dropout2d(dropout_rate)
        self.dropout_2 = nn.Dropout2d(dropout_rate)
        
        # Attention map generation
        self.conv2 = nn.Conv2d(intermediate_channels, 1, 3, 1, 1, bias=False)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x_att = self.conv1_rep(x)
        x_mod = x_att 
        x_mod = self.dropout_1(x_mod)  
        
        attention_map = self.conv2(x_mod)
        attention_map = self.act2(attention_map)
        output = x * attention_map
        output = self.dropout_2(output)
        
        return output
    
    def switch_to_deploy(self):
        """GSA 내부의 RepConv를 융합"""
        self.conv1_rep.switch_to_deploy()
 
# Stem
class Stem(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv1 = RepConv(i, o//2, 3, 2, 1, use_identity=True)
        self.conv2 = RepConv(o//2, o, 3, 1, 1, use_identity=True)
   
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 
class Downsample(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = RepConv(i, o, 3, 2, 1, use_identity=True)
   
    def forward(self, x):
        return self.conv(x)

class DWConvBlock(nn.Module):
    def __init__(self, in_dim, expansion_ratio=5):
        super().__init__()
        hidden_channels = int(in_dim * expansion_ratio)
        self.dw = RepConv(in_dim, in_dim, 7, 1, 3, groups=in_dim, use_identity=True)
        self.fc1 = RepConv(in_dim, hidden_channels, 1, 1, 0, use_identity=True, use_activation=True)
        self.fc2 = RepConv(hidden_channels, in_dim, 1, 1, 0, use_identity=True, use_activation=False)
    def forward(self, x):
        out = self.dw(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return x + out
 
# Backbone
class ProposedNet_Backbone(torch.nn.Module):
    def __init__(self, blocks, channels, emb_dims=512):
        super(ProposedNet_Backbone, self).__init__()
        self.stem = Stem(3, channels[0])
       
        self.stages = nn.ModuleList()
        in_channels = channels[0]
       
        for i in range(len(blocks)):
            stage_blocks = []
            stages = blocks[i]
            out_channels = channels[i]
           
            if i > 0:
                stage_blocks.append(Downsample(in_channels, out_channels))
           
            for _ in range(stages):
                stage_blocks.append(DWConvBlock(out_channels, expansion_ratio=5))
           
            self.stages.append(nn.Sequential(*stage_blocks))
            in_channels = out_channels
       
        # Gated Spatial Attention
        self.spatial_attention = GatedSpatialAttention(channels[1])
       
        self.expansion = RepConv(channels[-1], emb_dims, 1, 1, 0, use_identity=True, use_activation=True)
       
        self.model_init()
       
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
   
    def forward(self, inputs):
        # 저수준 특징 추출
        x = self.stem(inputs)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i==1:
                x = self.spatial_attention(x)
        x = self.expansion(x)
        return x
   
    def switch_to_deploy(self):
        """추론 모드로 전환: 모든 RepConv를 융합"""
        for module in self.modules():
            if isinstance(module, RepConv):
                module.switch_to_deploy()
            # GSA 내부의 RepConv도 융합
            elif isinstance(module, GatedSpatialAttention):
                module.switch_to_deploy()

class ProposedFERNet(nn.Module):
    def __init__(self, blocks, channels, emb_dims=512, num_classes=7):
        super(ProposedFERNet, self).__init__()
        self.blocks = blocks
        self.backbone = ProposedNet_Backbone(blocks, channels, emb_dims)
       
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(emb_dims, num_classes)
        )
   
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
   
    def switch_to_deploy(self):
        """추론 모드로 전환"""
        self.backbone.switch_to_deploy()

def ProposedNet(num_classes=7, **kwargs):
    return ProposedFERNet(
        blocks=[1, 4, 2],
        channels=[48, 96, 160],
        emb_dims=512,
        num_classes=num_classes
    )