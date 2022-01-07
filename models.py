from collections import OrderedDict
from subprocess import run
from pathlib import Path

import timm
import torch
from torch import nn
from torch.nn.parameter import Parameter
from future import *

DEBUG = False


def gem(x, p=1.5, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class GeM(nn.Module):
    def __init__(self, p=1.5, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def get_n_features(m):
    if hasattr(m, 'num_features'): return m.num_features            
    if hasattr(m, 'final_conv'  ): return m.final_conv.out_channels
    if hasattr(m, 'head'        ): return m.head.in_features       
    if hasattr(m, 'classifier'  ): return m.classifier.in_features 
    if hasattr(m, 'fc'          ): return m.fc.in_features         
    raise NotImplementedError(f"unkown model type:\n{m}")

def skip_head(m):
    for attr in ['classifier', 'global_pool', 'head', 'fc']:
        if hasattr(m, attr): setattr(m, attr, nn.Identity())
    return m

def compare_state_dicts(a, b, check_shapes=False):
    missing_in_a = [k for k in b if not k in a]
    missing_in_b = [k for k in a if not k in b]
    for k in missing_in_a:
        print(f"{' ':40}{k:40}")
    for k in missing_in_b:
        print(f"{k:40}{' ':40}")
    if check_shapes:
        for k in set(a.keys()).intersection(set(b.keys())):
            if a[k].shape != b[k].shape:
                print(f"size mismatch in {k:40} {str(list(a[k].shape)):12} {str(list(b[k].shape)):12}")

def get_pretrained_model(cfg):
    """Initialize pretrained_model for a new fold based on global variables
    
    Only fold-dependent variables must be passed in."""
    # AdaptiveMaxPool2d does not work with xla, use pmp = concat_pool = False
    pretrained = (cfg.rst_name is None)    
    pooling_layer = GeM() if cfg.use_gem else nn.AdaptiveAvgPool2d(output_size=1)
    body = timm.create_model(cfg.arch_name, pretrained=pretrained)
    n_features = get_n_features(body)
    body = skip_head(body)

    # Try vanilla head w/o bottleneck, BN, and dropout (like efnb7 models)
    if cfg.arch_name.startswith('cait'):
        head = nn.Sequential(nn.Dropout(p=cfg.dropout_ps[0]),
                             nn.Linear(n_features, cfg.n_classes))
    else:
        head = nn.Sequential(pooling_layer,
                             nn.Flatten(),
                             nn.Dropout(p=cfg.dropout_ps[0]),
                             nn.Linear(n_features, cfg.n_classes))

    pretrained_model = nn.Sequential(OrderedDict([('body', body), ('head', head)]))

    if cfg.rst_name:
        # Dont ever use xser: stores each tensor in a separate file!
        rst_file = Path(cfg.rst_path)/f'{removesuffix(cfg.rst_name, ".pth")}.pth'
        state_dict = torch.load(rst_file, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        if keys[0].startswith('0.') and keys[-1].startswith('1.'):
            # Fastai model: rename body keys, skip head if head != 'head'
            head = 'skip_head'
            print(f"Fastai model, renaming keys in state_dict: '0'/'1' -> 'body'/'{head}'")
            for k in keys:
                k_new = 'body' + k[1:] if k[0] == '0' else head
                state_dict[k_new] = state_dict.pop(k)
        if keys[0].startswith('model'):
            # Chest14-pretrained model from siimnihpretrained
            for key in keys:
                if key.startswith('model.'):
                    new_key = key.replace('model.', 'body.')
                    state_dict[new_key] = state_dict.pop(key)
                elif cfg.use_gem and key == 'pooling.p':
                    state_dict['head.0.p'] = state_dict.pop(key)
                    print(f'Pretrained weights: found pooling.p = {state_dict["head.0.p"].item()}')
                else:
                    print(f'Pretrained weights: skipping {key}')
                    _ = state_dict.pop(key)
            keys = list(state_dict.keys())
        if 'pretrained' in cfg.rst_path:
            for k in keys:
                if k.startswith('head') and (k.endswith('weight') or k.endswith('bias')):
                    v = state_dict.pop(k)
                    print(f'Pretrained weights: skipping {k} {list(v.size())}')
        try:
            incomp = pretrained_model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            compare_state_dicts(pretrained_model.state_dict(), state_dict, check_shapes=True)
            raise
        print(f"Restarting training from {rst_file}")
        if len(incomp) == 2: 
            print(f"Loaded state_dict has {len(incomp[0])} missing keys, {len(incomp[1])} unexpected keys.")
            if len(incomp[0]) + len(incomp[1]) > 2: 
                compare_state_dicts(pretrained_model.state_dict(), state_dict)            

    # change BN running average parameters
    for n, m in pretrained_model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            print("Setting BN eps, momentum")
            m.eps      = cfg.bn_eps
            m.momentum = cfg.bn_momentum

    # EfficientNet-V2 body has SiLU, BN (everywhere), but no Dropout.
    # Fused-MBConv (stages 1-3) and SE + 3x3-group_conv (group_size=1, stages 4-7).
    #print("FC stats:")
    #print("weight:", pretrained_model.state_dict()['head.3.weight'].mean(), pretrained_model.state_dict()['head.3.weight'].std())
    #print("bias:  ", pretrained_model.state_dict()['head.3.bias'].mean(), pretrained_model.state_dict()['head.3.bias'].std())    
    return pretrained_model

def get_smp_model(cfg):
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError:
        run('pip install -qU git+https://github.com/qubvel/segmentation_models.pytorch'.split(), capture_output=True)
        import segmentation_models_pytorch as smp
    print("[ √ ] segmentation_models_pytorch:", smp.__version__)

    pretrained_model = smp.DeepLabV3Plus(
        encoder_name=f'tu-{cfg.arch_name}',           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_depth=5,
        encoder_weights=None if cfg.rst_name else "imagenet",
        encoder_output_stride=16,
        decoder_atrous_rates=[12, 24, 36],
        decoder_channels=256,                     # segmentation_head bottleneck
        in_channels=3,                            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                                # model output channels (number of classes in your dataset)
        aux_params=dict(                          # for classifier head
            classes=cfg.n_classes,
            pooling="avg",
            dropout=cfg.dropout_ps[0],
            activation=None,
        ),
    )
    
    if cfg.add_hidden_layer:
        nf = 1280
        pooling_layer = GeM() if cfg.use_gem else timm.models.layers.SelectAdaptivePool2d(pool_type='avg', flatten=True)
        if not isinstance(pretrained_model.classification_head[0], nn.AdaptiveAvgPool2d):
            print(f"Warning: replacing {pretrained_model.classification_head[0]} -> hidden_layer")
        else:
            print(f"Adding efficientnet hidden_layer(256, {nf}) to classifier head.")
        pretrained_model.classification_head = nn.Sequential(OrderedDict(
            conv_head = nn.Conv2d(256, nf, kernel_size=(1, 1), stride=(1, 1), bias=False),
            #bn2 = nn.BatchNorm2d(nf, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            act2 = nn.SiLU(inplace=True),
            global_pool = pooling_layer,
            dropout = nn.Dropout(p=cfg.dropout_ps[0]),
            classifier = nn.Linear(in_features=nf, out_features=cfg.n_classes, bias=True),
        ))
    
    if cfg.rst_name:
        rst_file = Path(cfg.rst_path)/f'{removesuffix(cfg.rst_name, ".pth")}.pth'
        state_dict = torch.load(rst_file, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        print(f"{cfg.rst_name} keys:", keys[0])
        if keys[0].startswith('0.') and keys[-1].startswith('1.'):
            # Fastai model: rename body keys, skip head if head != 'head'
            head = 'skip_head'
            print(f"Fastai model, renaming keys in state_dict: '0'/'1' -> 'encoder.model'/'{head}'")
            for k in keys:
                k_new = 'encoder.model' + k[1:] if k[0] == '0' else head
                state_dict[k_new] = state_dict.pop(k)
        if keys[0].startswith('model'):
            # Chest14-pretrained model from siimnihpretrained
            for key in keys:
                if key.startswith('model.'):
                    new_key = key.replace('model.', 'encoder.model.')
                    state_dict[new_key] = state_dict.pop(key)
                elif cfg.use_gem and key == 'pooling.p':
                    state_dict['head.0.p'] = state_dict.pop(key)
                    print(f'Pretrained weights: found pooling.p = {state_dict["head.0.p"].item()}')
                else:
                    print(f'Pretrained weights: skipping {key}')
                    _ = state_dict.pop(key)
            keys = list(state_dict.keys())
        if 'pretrained' in cfg.rst_path or 'siimcovid-classifier-rst' in cfg.rst_path:
            for k in keys:
                if 'efficientnet' in cfg.arch_name and (k.startswith('body.conv_head') or k.startswith('body.bn2')):
                    if cfg.add_hidden_layer and nf == 1280:
                        new_key = k.replace('body', 'classification_head')
                        state_dict[new_key] = state_dict.pop(k)
                    else:
                        _ = state_dict.pop(k)  # smp removes conv_head from efficientnet encoders
                        print(f'[ √ ] Pretrained weights: skipping {k}')
                elif k.startswith('body'):
                    new_key = k.replace('body.', 'encoder.model.')
                    new_key = new_key.replace('stem.', 'stem_')
                    new_key = new_key.replace('conv_stem_weight', 'conv_stem.weight')
                    new_key = new_key.replace('stages.', 'stages_')
                    state_dict[new_key] = state_dict.pop(k)
                elif k.startswith('head') and (k.endswith('weight') or k.endswith('bias')):
                    v = state_dict.pop(k)
                    print(f'[ √ ] Pretrained weights: skipping {k} {list(v.size())}')
                elif cfg.use_gem and k == 'head.0.p':
                    new_key = 'classification_head.' + ('global_pool.p' if cfg.add_hidden_layer else '0.p')
                    state_dict[new_key] = state_dict.pop(k)
                else:
                    print(f'[ √ ] Pretrained weights: skipping {k}')
                    _ = state_dict.pop(k)
        try:
            incomp = pretrained_model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            compare_state_dicts(pretrained_model.state_dict(), state_dict, check_shapes=True)
            raise
        print(f"[ √ ] Restarting training from {rst_file}")
        if len(incomp) == 2:
            missing, unexpected = incomp
            missing = [k for k in missing if not (k.startswith('decoder') or k.startswith('seg'))]
            print(f"[ √ ] Loaded state_dict has {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
            if len(missing) + len(unexpected) > 2: 
                compare_state_dicts(pretrained_model.state_dict(), state_dict)
    
    # change BN running average parameters
    for n, m in pretrained_model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if DEBUG: print("Setting BN eps, momentum")
            m.eps      = cfg.bn_eps
            m.momentum = cfg.bn_momentum
    
    return pretrained_model