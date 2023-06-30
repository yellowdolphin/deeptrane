from collections import OrderedDict
from subprocess import run
from pathlib import Path
import math
from functools import partial

import timm
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from future import removesuffix

DEBUG = False


class AdaptiveCatPool2d(nn.Module):
    "Layer that concatenates `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, output_size=None):
        self.output_size = output_size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def gem(x, p=1.5, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=1.5, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class ArcModule(nn.Module):
    # from https://github.com/pudae/kaggle-humpback/blob/master/tasks/identifier.py
    def __init__(self, in_features, out_features, s=65, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).to(labels.device)
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

    def __repr__(self):
        return f'{self.__class__.__name__}('\
               f'in_features={self.in_features}, out_features={self.out_features}, '\
               f's={self.s}, m={self.m})'


class ArcMarginProduct(nn.Module):
    """
    https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.30, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(input, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def __repr__(self):
        return f'{self.__class__.__name__}('\
               f'in_features={self.in_features}, out_features={self.out_features}, '\
               f's={self.s}, m={self.m}, easy_margin={self.easy_margin}, ls_eps={self.ls_eps})'


class AddMarginProduct(nn.Module):
    """
    https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(input, F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=label.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return f'{self.__class__.__name__}('\
               f'in_features={self.in_features}, out_features={self.out_features}, '\
               f's={self.s}, m={self.m})'


class ArcNet(nn.Module):
    "Wraps a pretrained model with ArcFace head, forward requires labels."
    def __init__(self, cfg, model):
        super().__init__()

        self.model = model
        in_features = cfg.channel_size or 512
        out_features = cfg.n_classes
        s, m = cfg.arcface_s, cfg.arcface_m
        self.arc = (
            ArcModule(in_features, out_features, s=s, m=m) if cfg.arcface == 'ArcModule' else
            ArcMarginProduct(in_features, out_features, s=s, m=m) if cfg.arcface == 'ArcMarginProduct' else
            AddMarginProduct(in_features, out_features, s=s, m=m) if cfg.arcface == 'AddMarginProduct' else
            nn.Identity())
        self.requires_labels = True

        if hasattr(self.model, 'mean') and hasattr(self.model, 'std'):
            print(f"ArcNet: pretrained model has mean {self.model.mean:.5f}, std {self.model.std:.5f}")
        if hasattr(self.model, 'input_range') and hasattr(self.model, 'input_space'):
            print(f"ArcNet: pretrained model has input_range {self.model.input_range}, input_space {self.model.input_space}")
        self.mean = self.model.mean if hasattr(self.model, 'mean') else [0.5, 0.5, 0.5]
        self.std = self.model.std if hasattr(self.model, 'std') else [0.5, 0.5, 0.5]
        self.input_range = self.model.input_range if hasattr(self.model, 'input_range') else [0, 1]
        self.input_space = self.model.input_space if hasattr(self.model, 'input_space') else 'RGB'
        self.valid_mode = 'normal' if cfg.deotte else 'embedding'

    def forward(self, images, labels=None):
        # In validation, forward is called w/o labels.
        # Hence, output features have shape [N, 512] rather than [N, n_classes].
        # But task.loss(features, labels) is called the same way!
        # As task.loss is same in train/valid, dataloader is pytorch one,
        # the dataset must provide some labels in the 0...511 range that represent ...what?
        # Both task.forward() and task.inference() are called w (train) or w/o (valid) labels!
        # If inference is called with outputs, the latter are just returned, otherwise same as forward.
        # When called w/o labels, the features are normalized but not passed through ArcModule,
        # hence have size [N, 512]!
        if (self.valid_mode == 'embedding' and not self.training) or (labels is None):
            return self.model(images)  # skip normalize, return embeddings
        features = F.normalize(self.model(images))
        return self.arc(features, labels)


def is_bn(name):
    return any((name.startswith('bn1.'), name.startswith('bn2.'), name.startswith('bn3.'),
                '.bn1.' in name, '.bn2.' in name, '.bn3.' in name))


def get_n_features(m, default='raise'):
    if hasattr(m, 'num_features'): return m.num_features
    if hasattr(m, 'final_conv'  ): return m.final_conv.out_channels
    if hasattr(m, 'head'        ): return m.head.in_features
    if hasattr(m, 'classifier'  ): return m.classifier.in_features
    if hasattr(m, 'fc'          ): return m.fc.in_features
    if default == 'raise':
        raise NotImplementedError(f"unkown model type:\n{m}")
    return default


def get_out_features(m, default='raise'):
    if hasattr(m, 'out_features'): return m.out_features
    if hasattr(m, 'out_channels'): return m.out_channels
    if default == 'raise':
        raise NotImplementedError(f"unkown model type:\n{m}")
    return default


def get_last_out_features(m, default='raise'):
    if not hasattr(m, 'children'): return get_out_features(m, default)
    out_features = [get_out_features(c, None) for c in m.children()]
    out_features = [n for n in out_features if n]
    if out_features: return out_features[-1]
    if default == 'raise':
        raise NotImplementedError(f"unkown model type:\n{m}")
    return default


def body_forward(self, x):
    "Replaces model.forward in timm models"
    return self.forward_features(x)


def skip_head(m):
    #if hasattr(m, 'forward_features'):
    #    # Monkey-patch timm model to skip model.forward_head() call
    #    print("Monkey-patching timm model.forward:")
    #    print(m.forward.__code__)
    #    m.forward = partial(body_forward, m)
    #    return m

    for attr in ['classifier', 'global_pool', 'head', 'fc']:
        if hasattr(m, attr): setattr(m, attr, nn.Identity())
    return m


def compare_state_dicts(a, b, check_shapes=False):
    """a, b may be state_dicts or just their keys.
    
    `check_shapes` requires state_dicts."""
    missing_in_a = [k for k in b if k not in a]
    missing_in_b = [k for k in a if k not in b]
    for k in missing_in_a:
        print(f"{' ':40}{k:40}")
    for k in missing_in_b:
        print(f"{k:40}{' ':40}")

    if check_shapes:
        for k in set(a.keys()).intersection(set(b.keys())):
            if a[k].shape != b[k].shape:
                print(f"size mismatch in {k:40} {str(list(a[k].shape)):12} {str(list(b[k].shape)):12}")


def get_bottleneck_params(cfg):
    "Define one Dropout (maybe zero) per FC + optional final Dropout"
    dropout_ps = cfg.dropout_ps or []
    lin_ftrs = cfg.lin_ftrs or []
    if len(dropout_ps) > len(lin_ftrs) + 1:
        raise ValueError(f"too many dropout_ps ({len(dropout_ps)}) for {len(lin_ftrs)} lin_ftrs")
    final_dropout = dropout_ps.pop() if len(dropout_ps) == len(lin_ftrs) + 1 else 0
    num_missing_ps = len(lin_ftrs) - len(dropout_ps)
    dropout_ps.extend([0] * num_missing_ps)

    return lin_ftrs, dropout_ps, final_dropout


def get_pretrained_model(cfg):
    """Initialize pretrained_model for a new fold based on global variables

    Only fold-dependent variables must be passed in.
    bottleneck: callable that returns an nn.Sequential instance"""
    # AdaptiveMaxPool2d does not work with xla, use pmp = concat_pool = False
    pretrained = (cfg.rst_name is None) and 'defaults' not in cfg.tags
    body = timm.create_model(cfg.arch_name, pretrained=pretrained)
    n_features = get_n_features(body)
    body = skip_head(body)
    print(n_features, "features after last Conv")

    # Pooling
    if cfg.get_pooling is not None:
        pooling_layer = cfg.get_pooling(cfg, n_features)
    elif cfg.pool == 'flatten':
        pooling_layer = []
    elif cfg.pool == 'fc':
        assert cfg.feature_size, 'cfg.pool=fc needs cfg.feature_size'
        pooling_layer = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(n_features * cfg.feature_size, n_features),
            nn.BatchNorm1d(n_features))
    elif cfg.pool == 'gem':
        pooling_layer = GeM()
    elif cfg.pool in ['cat', 'concat']:
        pooling_layer = AdaptiveCatPool2d(output_size=1)
    elif cfg.pool == 'max':
        pooling_layer = nn.AdaptiveMaxPool2d(output_size=1)
    else:
        pooling_layer = nn.AdaptiveAvgPool2d(output_size=1)
    if cfg.pool in ['max', 'concat'] and cfg.xla:
        print("WARNING: MaxPool not supported by torch_xla")
    n_features = get_last_out_features(pooling_layer, None) or n_features
    print(n_features, "features after pooling")

    # Bottleneck(s)
    if cfg.get_bottleneck is not None:
        bottleneck = cfg.get_bottleneck(cfg, n_features)
    else:
        lin_ftrs, dropout_ps, final_dropout = get_bottleneck_params(cfg)
        act_head = getattr(nn, cfg.act_head) if isinstance(cfg.act_head, str) else cfg.act_head
        bottleneck = []
        for p, out_features in zip(dropout_ps, lin_ftrs):
            if p > 0: bottleneck.append(nn.Dropout(p=p))
            bottleneck.append(nn.Linear(n_features, out_features))
            if act_head: bottleneck.append(act_head())
            n_features = out_features
            if cfg.bn_head: bottleneck.append(nn.BatchNorm1d(n_features))
        if final_dropout:
            bottleneck.append(nn.Dropout(p=final_dropout))

    n_features = get_last_out_features(bottleneck, None) or n_features
    print(n_features, "features after bottleneck")

    if cfg.deotte:
        # Happywhale model used in my final TF notebooks
        head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                             #nn.BatchNorm2d(n_features),   # also try BN1d after Flatten
                             nn.Flatten(),
                             nn.BatchNorm1d(n_features),
                             )
        cfg.channel_size = n_features
    elif cfg.feature_size:
        # Model used by pudae (Humpback-whale-identification) with FC instead of AvgPool
        head = nn.Sequential(nn.BatchNorm2d(n_features),
                             nn.Dropout2d(p=cfg.dropout_ps[0]),
                             nn.Flatten(),
                             nn.Linear(n_features * cfg.feature_size, cfg.channel_size),
                             nn.BatchNorm1d(cfg.channel_size))
    elif cfg.arch_name.startswith('cait'):
        head = nn.Sequential(*bottleneck,
                             nn.Linear(n_features, cfg.channel_size or cfg.n_classes))
    else:
        print(f"building output layer {n_features, cfg.channel_size or cfg.n_classes}")
        head = nn.Sequential(pooling_layer,
                             nn.Flatten(),
                             *bottleneck,
                             nn.Linear(n_features, cfg.channel_size or cfg.n_classes))

    pretrained_model = nn.Sequential(OrderedDict([('body', body), ('head', head)]))

    if cfg.arcface:
        pretrained_model = ArcNet(cfg, pretrained_model)
        if DEBUG:
            keys = list(pretrained_model.state_dict().keys())
            print("keys of ArcNet.state_dict:", keys[:2], '...', keys[-2:])

    if cfg.rst_name:
        # Dont ever use xser: stores each tensor in a separate file!
        rst_file = Path(cfg.rst_path) / f'{removesuffix(cfg.rst_name, ".pth")}.pth'
        state_dict = torch.load(rst_file, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        if DEBUG: print("keys of loaded state_dict:", keys[:2], '...', keys[-2:])
        if keys[0].startswith('0.') and keys[-1].startswith('1.'):
            # Fastai model: rename body keys, skip head if head != 'head'
            head = 'ignore_me'
            print(f"Fastai model, renaming keys in state_dict: '0'/'1' -> 'body'/'{head}'")
            for k in keys:
                k_new = 'body' + k[1:] if k[0] == '0' else head
                state_dict[k_new] = state_dict.pop(k)
        if cfg.modify_state_dict:
            # compatibility can be extended by assigning a translation function in the project module
            state_dict = cfg.modify_state_dict(state_dict, pretrained_model.state_dict())
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
    n_bn_layers = 0
    for n, m in pretrained_model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            n_bn_layers += 1
            m.eps = cfg.bn_eps
            m.momentum = cfg.bn_momentum
    if n_bn_layers:
        print(f"Setting eps, momentum for {n_bn_layers} BatchNorm layers")

    # EfficientNet-V2 body has SiLU, BN (everywhere), but no Dropout.
    # Fused-MBConv (stages 1-3) and SE + 3x3-group_conv (group_size=1, stages 4-7).
    #print("FC stats:")
    #print("weight:",
    #      pretrained_model.state_dict()['head.3.weight'].mean(),
    #      pretrained_model.state_dict()['head.3.weight'].std())
    #print("bias:  ",
    #      pretrained_model.state_dict()['head.3.bias'].mean(),
    #      pretrained_model.state_dict()['head.3.bias'].std())
    #print(pretrained_model)
    return pretrained_model


def get_output_layer(model):
    # obsolete, use model.get_classifier() instead
    if hasattr(model, 'head'):
        if isinstance(model.head, torch.nn.Linear):
            return model.head
        elif hasattr(model.head, 'fc'):
            assert isinstance(model.head.fc, torch.nn.Linear), f"weird (head.fc): {model.head.fc}"
            return model.head.fc
    elif hasattr(model, 'fc'):
        assert isinstance(model.fc, torch.nn.Linear), f"weird (fc): {model.fc}"
        return model.fc
    elif hasattr(model, 'classifier'):
        assert isinstance(model.classifier, torch.nn.Linear), f"weird (classifier): {model.classifier}"
        return model.classifier
    return None


def set_output_layer(model, layer):
    if hasattr(model, 'head'):
        if isinstance(model.head, torch.nn.Linear):
            model.head = layer
            return model.head
        elif hasattr(model.head, 'fc'):
            assert isinstance(model.head.fc, torch.nn.Linear), f"weird (head.fc): {model.head.fc}"
            model.head.fc = layer
            return model.head.fc
    elif hasattr(model, 'fc'):
        assert isinstance(model.fc, torch.nn.Linear), f"weird (fc): {model.fc}"
        model.fc = layer
        return model.fc
    elif hasattr(model, 'classifier'):
        assert isinstance(model.classifier, torch.nn.Linear), f"weird (classifier): {model.classifier}"
        model.classifier = layer
        return model.classifier
    return None


def get_pretrained_timm(cfg):
    """Initialize pretrained_model for a new fold based on cfg

    Use timm API as far as possible.

    Not sure: Custom pooling module can be assigned to cfg.pool, or factory to cfg.get_pooling,
    which is called with (cfg, n_features).

    Custom bottleneck factory can be assigned to cfg.get_bottleneck.
    """
    # AdaptiveMaxPool2d does not work with xla, use pmp = concat_pool = False
    pretrained = (cfg.rst_name is None) and 'defaults' not in cfg.tags
    pretrained_model = timm.create_model(cfg.arch_name, pretrained=pretrained,
                                         num_classes=cfg.n_classes or cfg.channel_size)

    pretrained_model.eval()
    inp = torch.randn(2, 3, 224, 224)
    unpooled_features = pretrained_model.forward_features(inp)
    embeddings = pretrained_model.forward_head(unpooled_features, pre_logits=True)
    predictions = pretrained_model.forward_head(unpooled_features)
    n_features = unpooled_features.shape[1]
    print(n_features, "features before pooling")

    # Pooling
    if cfg.pool is None:
        pooling_layer = None  # keep original
    elif cfg.get_pooling is not None:
        pooling_layer = cfg.get_pooling(cfg, n_features)
    elif cfg.pool == 'flatten':
        pooling_layer = []
    elif cfg.pool == 'fc':
        assert cfg.feature_size, 'cfg.pool=fc needs cfg.feature_size'
        pooling_layer = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(n_features * cfg.feature_size, n_features),
            nn.BatchNorm1d(n_features))
    elif cfg.pool == 'gem':
        pooling_layer = GeM()
    elif cfg.pool in ['cat', 'concat']:
        pooling_layer = AdaptiveCatPool2d(output_size=1)
    elif cfg.pool == 'max':
        pooling_layer = nn.AdaptiveMaxPool2d(output_size=1)
    else:
        pooling_layer = nn.AdaptiveAvgPool2d(output_size=1)
    if cfg.pool in ['max', 'concat'] and cfg.xla:
        print("WARNING: MaxPool not supported by torch_xla")
    #if pooling_layer is not None:
    #    pretrained_model.head.global_pool = pooling_layer
    #n_features = get_last_out_features(pooling_layer, None) or n_features
    #assert n_features == pretrained_model.forward
    #print(n_features, "features after pooling")

    if pooling_layer:
        n_features = torch.flatten(pooling_layer(unpooled_features), 1, -1).shape[1]
        print(f"{n_features} features after pooling")
    else:
        n_features = embeddings.shape[1]  # can be incorrect if custom pooling
        print(f"{n_features} features after original pooling")

    # improve initialization of output layer
    #set_output_layer(pretrained_model, nn.Linear(n_features, cfg.channel_size or cfg.n_classes))  # good: efn, rexnet
    if cfg.scale_output_layer is not None:
        if pooling_layer is not None:
            print("WARNING: scale_output_layer not supported with custom pooling")
        output_layer = pretrained_model.get_classifier()
        with torch.no_grad():
            fan_in = output_layer.weight.size(1)
            limit = math.sqrt(1 / fan_in) * cfg.scale_output_layer
            nn.init.uniform_(output_layer.weight, -limit, limit)
            nn.init.uniform_(output_layer.bias, -limit, limit)
            gain_w = output_layer.weight.detach().max().numpy() / math.sqrt(1 / fan_in)
            gain_b = output_layer.bias.detach().max().numpy() / math.sqrt(1 / fan_in)
            print("output layer weight gain:", gain_w)
            print("output layer bias gain:  ", gain_b)

    # Bottleneck(s)
    if cfg.get_bottleneck is not None:
        bottleneck = cfg.get_bottleneck(cfg, n_features)
    else:
        lin_ftrs, dropout_ps, final_dropout = get_bottleneck_params(cfg)
        act_head = getattr(nn, cfg.act_head) if isinstance(cfg.act_head, str) else cfg.act_head
        bottleneck = []
        for p, out_features in zip(dropout_ps, lin_ftrs):
            if p > 0: bottleneck.append(nn.Dropout(p=p))
            bottleneck.append(nn.Linear(n_features, out_features))
            if act_head: bottleneck.append(act_head())
            n_features = out_features
            if cfg.bn_head: bottleneck.append(nn.BatchNorm1d(n_features))
        if final_dropout:
            bottleneck.append(nn.Dropout(p=final_dropout))

    n_features = get_last_out_features(bottleneck, None) or n_features
    print(n_features, "features after bottleneck")

    if cfg.deotte:
        # Happywhale model used in my final TF notebooks
        head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                             #nn.BatchNorm2d(n_features),   # also try BN1d after Flatten
                             nn.Flatten(),
                             nn.BatchNorm1d(n_features),
                             )
        cfg.channel_size = n_features
    elif cfg.feature_size:
        # Model used by pudae (Humpback-whale-identification) with FC instead of AvgPool
        head = nn.Sequential(nn.BatchNorm2d(n_features),
                             nn.Dropout2d(p=cfg.dropout_ps[0]),
                             nn.Flatten(),
                             nn.Linear(n_features * cfg.feature_size, cfg.channel_size),
                             nn.BatchNorm1d(cfg.channel_size))
    elif pooling_layer:
        print(f"building output layer {n_features, cfg.channel_size or cfg.n_classes}")
        head = nn.Sequential(pooling_layer,
                             nn.Flatten(),
                             *bottleneck,
                             nn.Linear(n_features, cfg.channel_size or cfg.n_classes))
    else:
        head = None  # use original head, maybe replace FC by bottleneck

    class TimmWrapper(nn.Module):
        def __init__(self, timm_model, custom_head=None, bottleneck=None):
            super(TimmWrapper, self).__init__()
            self.timm_model = timm_model
            self.custom_head = custom_head
            self.bottleneck = bottleneck
            if self.custom_head:
                print("Custom head:")
                print(self.custom_head)
            elif self.bottleneck:
                print("Custom bottleneck:")
                print(self.bottleneck)

        def forward(self, x):
            unpooled_features = self.timm_model.forward_features(x)
            if self.custom_head is not None:
                return self.custom_head(unpooled_features)
            elif self.bottleneck is not None:
                embeddings = self.timm_model.forward_head(unpooled_features, pre_logits=True)
                return self.bottleneck(embeddings)
            return self.timm_model.forward_head(unpooled_features)

    if head or bottleneck:
        pretrained_model = TimmWrapper(pretrained_model, head, bottleneck)

    if cfg.arcface:
        pretrained_model = ArcNet(cfg, pretrained_model)
        if DEBUG:
            keys = list(pretrained_model.state_dict().keys())
            print("keys of ArcNet.state_dict:", keys[:2], '...', keys[-2:])

    if cfg.rst_name:
        # Dont ever use xser: stores each tensor in a separate file!
        rst_file = Path(cfg.rst_path) / f'{removesuffix(cfg.rst_name, ".pth")}.pth'
        state_dict = torch.load(rst_file, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        if DEBUG: print("keys of loaded state_dict:", keys[:2], '...', keys[-2:])
        if keys[0].startswith('0.') and keys[-1].startswith('1.'):
            # Fastai model: rename body keys, skip head if head != 'head'
            head = 'ignore_me'
            print(f"Fastai model, renaming keys in state_dict: '0'/'1' -> 'body'/'{head}'")
            for k in keys:
                k_new = 'body' + k[1:] if k[0] == '0' else head
                state_dict[k_new] = state_dict.pop(k)
        if cfg.modify_state_dict:
            # compatibility can be extended by assigning a translation function in the project module
            state_dict = cfg.modify_state_dict(state_dict, pretrained_model.state_dict())
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
    n_bn_layers = 0
    for n, m in pretrained_model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            n_bn_layers += 1
            m.eps = cfg.bn_eps
            m.momentum = cfg.bn_momentum
    if n_bn_layers:
        print(f"Setting eps, momentum for {n_bn_layers} BatchNorm layers")

    # either freeze or unfreeze head layers
    head = list(pretrained_model.children())[-1]
    if cfg.freeze_head:
        print("Freezing head layers:")
        print(head)
        for p in head.parameters():
            p.requires_grad = False
    else:
        for p in head.parameters():
            p.requires_grad = True

    # EfficientNet-V2 body has SiLU, BN (everywhere), but no Dropout.
    # Fused-MBConv (stages 1-3) and SE + 3x3-group_conv (group_size=1, stages 4-7).
    #print("FC stats:")
    #print("weight:",
    #      pretrained_model.state_dict()['head.3.weight'].mean(),
    #      pretrained_model.state_dict()['head.3.weight'].std())
    #print("bias:  ",
    #      pretrained_model.state_dict()['head.3.bias'].mean(),
    #      pretrained_model.state_dict()['head.3.bias'].std())
    return pretrained_model


class PoolCat(nn.Module):
    "This Pooling layer connects a timm FeatureListNet to a custom head/bottleneck"
    def __init__(self, pool='avg'):
        print(f"Pooling: {pool.__name__ if isinstance(pool, nn.Module) else pool}")
        super().__init__()
        self.pool = (
            pool() if isinstance(pool, nn.Module) else
            pool if hasattr(pool, '__call__') else
            nn.AdaptiveAvgPool2d(output_size=1) if (pool.lower() == 'avg') else
            nn.AdaptiveMaxPool2d(output_size=1) if (pool.lower() == 'max') else
            torch.flatten if (pool.lower() == 'flatten') else
            GeM() if (pool.lower() == 'gem') else
            AdaptiveCatPool2d(output_size=1) if pool.lower() in ['cat', 'concat'] else
            partial(getattr(F, pool.lower()), output_size=1) if hasattr(F, pool.lower()) else
            getattr(torch.nn, pool)(output_size=1) if hasattr(torch.nn, pool) else None)
        if self.pool is None: raise ValueError(f'{pool} is no supported pooling option. '
            'Supported are: "avg" (default) "max", None, "flatten", "cat", "gem", '
            'any callable, nn.Module class (or class name), any attr (str) from nn.functional. '
            'If cfg.pool is None, the original pooling (last pooling in original model) is used.')

    def forward(self, input_list):
        pooled = [self.pool(x) for x in input_list]
        return torch.cat(pooled, dim=1)


def last_n_feature_indices(arch_name, n_features):
    """Return (hopefully) valid list `out_indices` for timm.create_model().
    
    Some timm models don't have feature_info, others don't accept negative out_indices.
    Try first, fallback to second to get out_indices for the last n_features features."""
    model = timm.create_model(arch_name, pretrained=False)
    if hasattr(model, 'feature_info'):
        feature_index = [i for i, _ in enumerate(model.feature_info)]
        return feature_index[-n_features:]
    return list(range(-n_features, 0))


def adjust_head_drop(model, dropout_p):
    dropout_p = float(dropout_p)
    if hasattr(model, 'head_drop'):
        model.head_drop.p = dropout_p
    elif hasattr(model, 'head') and hasattr(model.head, 'drop'):
        model.head.drop.p = dropout_p
    else:
        print("Warning: no head_drop or head.drop found")


def get_pretrained_timm2(cfg):
    """Initialize pretrained_model for a new fold based on cfg

    Use timm API as far as possible.

    Not sure: Custom pooling module can be assigned to cfg.pool, or factory to cfg.get_pooling,
    which is called with (cfg, n_features).

    Custom bottleneck factory can be assigned to cfg.get_bottleneck.
    """
    pretrained = (cfg.rst_name is None) and 'defaults' not in cfg.tags
    n_features = cfg.n_features or 1  # last n features to pool and concat from body
    act_head = getattr(nn, cfg.act_head) if isinstance(cfg.act_head, str) else cfg.act_head
    try:
        body = timm.create_model(cfg.arch_name, pretrained=pretrained, features_only=True,
                                 out_indices=last_n_feature_indices(cfg.arch_name, n_features))
        add_head = True
        in_features = sum(body.feature_info[-i]['num_chs'] for i in range(1, n_features + 1))
    except (RuntimeError, AttributeError):
        print(f"Warning: {cfg.arch_name} does not support the features_only kwarg.")
        if cfg.lin_ftrs:
            if cfg.feature_size:
                print("    feature_size ({cfg.feature_size}) will be ignored.")
                cfg.feature_size = None
            in_features, head_dropout_p = cfg.lin_ftrs.pop(0), cfg.dropout_ps.pop(0)
            body = timm.create_model(cfg.arch_name, pretrained=pretrained, num_classes=in_features)
            adjust_head_drop(body, head_dropout_p)
            print("    Adjusting original head:")
            print(body.get_classifier())
            add_head = True
        else:
            print("    Only the classifier will be adapted.")
            #assert not cfg.lin_ftrs, f"cfg.lin_ftrs not supported by {cfg.arch_name}"
            add_head = False
            in_features = None

    # Pooling
    if cfg.get_pooling is not None:
        pooling_layer = cfg.get_pooling(cfg, in_features)
    elif add_head and body.__class__.__name__ != 'FeatureListNet':
        # pack optional head layers after first FC into "pooling_layer"
        pooling_layer = nn.Sequential()
        if act_head: pooling_layer.append(act_head())
        if cfg.bn_head: pooling_layer.append(nn.BatchNorm1d(in_features))
    elif add_head:
        if cfg.pool is None:
            # try to get pooling from original model head
            pool_layers = [k for k, v in timm.create_model(cfg.arch_name).named_modules() if 'pool' in k]
            if pool_layers:
                cfg.pool = pool_layers[-1]
        pooling_layer = PoolCat(cfg.pool)
    if cfg.feature_size: in_features *= cfg.feature_size  # unpooled H * W
    if in_features: print(f"features after pooling: {in_features}")

    # Bottleneck(s)
    if cfg.get_bottleneck is not None:
        bottleneck = cfg.get_bottleneck(cfg, in_features)
    elif add_head:
        lin_ftrs, dropout_ps, final_dropout = get_bottleneck_params(cfg)
        bottleneck = []
        for p, out_features in zip(dropout_ps, lin_ftrs):
            if p > 0: bottleneck.append(nn.Dropout(p=p))
            bottleneck.append(nn.Linear(in_features, out_features))
            in_features = out_features
            if act_head: bottleneck.append(act_head())
            if cfg.bn_head: bottleneck.append(nn.BatchNorm1d(in_features))
        if final_dropout:
            bottleneck.append(nn.Dropout(p=final_dropout))
    if in_features: print(f"features after bottleneck: {in_features}")

    # Compose Head
    if add_head:
        output_layer = nn.Linear(in_features, cfg.channel_size or cfg.n_classes, bias=True)
        head = nn.Sequential(
            pooling_layer,
            torch.nn.Flatten(),
            *bottleneck,
            output_layer)

    # Compose Model
    pretrained_model = (
        nn.Sequential(OrderedDict([('body', body), ('head', head)])) if add_head else
        timm.create_model(cfg.arch_name, pretrained, num_classes=cfg.channel_size or cfg.n_classes))

    # Scaled initialization of output layer
    if add_head and (cfg.scale_output_layer != 1):
        output_layer = pretrained_model.head[-1]
        with torch.no_grad():
            fan_in = output_layer.weight.size(1)
            limit = math.sqrt(1 / fan_in) * cfg.scale_output_layer
            nn.init.uniform_(output_layer.weight, -limit, limit)
            nn.init.uniform_(output_layer.bias, -limit, limit)
            gain_w = output_layer.weight.detach().max().numpy() / math.sqrt(1 / fan_in)
            gain_b = output_layer.bias.detach().max().numpy() / math.sqrt(1 / fan_in)
            print("output layer weight gain:", gain_w)
            print("output layer bias gain:  ", gain_b)

    # Extra model wrappers
    if cfg.arcface:
        pretrained_model = ArcNet(cfg, pretrained_model)
        if DEBUG:
            keys = list(pretrained_model.state_dict().keys())
            print("keys of ArcNet.state_dict:", keys[:2], '...', keys[-2:])

    # Restart training
    if cfg.rst_name:
        # Dont ever use xser: stores each tensor in a separate file!
        rst_file = Path(cfg.rst_path) / f'{removesuffix(cfg.rst_name, ".pth")}.pth'
        state_dict = torch.load(rst_file, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        if DEBUG: print("keys of loaded state_dict:", keys[:2], '...', keys[-2:])
        if keys[0].startswith('0.') and keys[-1].startswith('1.'):
            # Fastai model: rename body keys, skip head if head != 'head'
            head = 'ignore_me'
            print(f"Fastai model, renaming keys in state_dict: '0'/'1' -> 'body'/'{head}'")
            for k in keys:
                k_new = 'body' + k[1:] if k[0] == '0' else head
                state_dict[k_new] = state_dict.pop(k)
        if cfg.modify_state_dict:
            # compatibility can be extended by defining this function in the project module
            state_dict = cfg.modify_state_dict(state_dict, pretrained_model.state_dict())
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

    # Change BN running average parameters
    n_bn_layers = 0
    for m in pretrained_model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            n_bn_layers += 1
            m.eps = cfg.bn_eps
            m.momentum = cfg.bn_momentum
    if n_bn_layers:
        print(f"Setting eps, momentum for {n_bn_layers} BatchNorm layers")

    # Either freeze or unfreeze head layers
    head = pretrained_model.head if hasattr(pretrained_model, 'head') else pretrained_model.get_classifier()
    if cfg.freeze_head:
        print("Freezing head layers:")
        print(head)
        for p in head.parameters():
            p.requires_grad = False
    else:
        for p in head.parameters():
            p.requires_grad = True

    return pretrained_model


def get_smp_model(cfg):
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError:
        run('pip install -qU git+https://github.com/qubvel/segmentation_models.pytorch'.split())
        import segmentation_models_pytorch as smp
    print("[ √ ] segmentation_models_pytorch:", smp.__version__)

    pretrained_model = smp.DeepLabV3Plus(
        encoder_name=f'tu-{cfg.arch_name}',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_depth=5,
        encoder_weights=None if cfg.rst_name else "imagenet",
        encoder_output_stride=16,
        decoder_atrous_rates=[12, 24, 36],
        decoder_channels=256,                # segmentation_head bottleneck
        in_channels=3,                       # 1 for gray-scale images, 3 for RGB
        classes=1,                           # model output channels
        aux_params=dict(                     # for classifier head
            classes=cfg.n_classes,
            pooling="avg",
            dropout=cfg.dropout_ps[0],
            activation=None,
        ),
    )

    if cfg.add_hidden_layer:
        nf = 1280
        pooling_layer = (GeM() if cfg.use_gem else
                         timm.models.layers.SelectAdaptivePool2d(pool_type='avg', flatten=True))
        if not isinstance(pretrained_model.classification_head[0], nn.AdaptiveAvgPool2d):
            print(f"Warning: replacing {pretrained_model.classification_head[0]} -> hidden_layer")
        else:
            print(f"Adding efficientnet hidden_layer(256, {nf}) to classifier head.")
        pretrained_model.classification_head = nn.Sequential(OrderedDict(
            conv_head=nn.Conv2d(256, nf, kernel_size=(1, 1), stride=(1, 1), bias=False),
            #bn2=nn.BatchNorm2d(nf, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            act2=nn.SiLU(inplace=True),
            global_pool=pooling_layer,
            dropout=nn.Dropout(p=cfg.dropout_ps[0]),
            classifier=nn.Linear(in_features=nf, out_features=cfg.n_classes, bias=True),
        ))

    if cfg.rst_name:
        rst_file = Path(cfg.rst_path) / f'{removesuffix(cfg.rst_name, ".pth")}.pth'
        state_dict = torch.load(rst_file, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        print(f"{cfg.rst_name} keys:", keys[0])
        if keys[0].startswith('0.') and keys[-1].startswith('1.'):
            # Fastai model: rename body keys, skip head if head != 'head'
            head = 'ignore_me'
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
            m.eps = cfg.bn_eps
            m.momentum = cfg.bn_momentum

    return pretrained_model
