import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import loss


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
    def __init__(self, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))

        return sum(losses)


class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel_s, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) 
            for channel_s, channel in zip(channels_s, channels_t)
        ])

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
       
        return sum(losses)

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


class ATLoss(nn.Module):
    """ Attention Transfer (attention maps) — Zagoruyko & Komodakis 2017 """
    def __init__(self, eps=1e-6):
        super(ATLoss, self).__init__()
        self.eps = eps
        
    @staticmethod
    def _attention(f, eps):
        att = f.pow(2).mean(1, keepdim=True) # (N, 1, H, W)
        att = att.flatten(2) # (N, 1, H*W)
        att = att / (att.norm(p=2, dim=2, keepdim=True) + eps) # (N, 1, H*W)
        return att
    
    def forward(self, y_s, y_t):
        loss = 0.0
        for feat_s, feat_t in zip(y_s, y_t):
            As = self._attention(feat_s, self.eps)
            At = self._attention(feat_t, self.eps)
            loss += F.mse_loss(As, At, reduction='mean')
        return loss


class SKDLoss(nn.Module):
    """Spatial-Knowledge-Distillation (Thin-Plate, 2021)(memory-friendly)"""
    def __init__(self, k=256, normalize=True):
        super(SKDLoss, self).__init__()
        self.k = k
        self.mse = nn.MSELoss(reduction='mean')
        self.normalize = normalize
    
    def _corr_sampling(self, feat):
        n, c, h, w = feat.shape
        feat = feat.flatten(2).contiguous() # (N, C, H*W)
        if self.normalize:
            feat = F.normalize(feat, dim=1)
        hw = h * w
        
        if self.k < hw:
            idx = torch.randperm(hw, device=feat.device)[:self.k]
            feat = feat[..., idx]  # (N, C, K)
        S = torch.matmul(feat.transpose(1, 2), feat) / c # (N, K, K)
        return S
    
    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        for feat_s, feat_t in zip(y_s, y_t):
            S_s = self._corr_sampling(feat_s)
            S_t = self._corr_sampling(feat_t.detach())
            losses.append(self.mse(S_s, S_t))
            
        return torch.stack(losses).mean()


class PKDLoss(nn.Module):
    """PKD: Pearson-Correlation based KD  (CVPR-21)."""
    def __init__(self, normalize=True, resize_stu=True):
        super(PKDLoss, self).__init__()
        self.resize_stu = resize_stu
        self.normalize = normalize
        self.mse = nn.MSELoss(reduction='mean')
    
    @staticmethod
    def _feature_norm(x):
        n, c, h, w = x.shape
        x = x.permute(1, 0, 2, 3).contiguous().view(c, -1)   # (C, N*H*W)
        mu  = x.mean(dim=-1, keepdim=True)
        std = x.std (dim=-1, keepdim=True)
        x = ((x - mu) / (std + 1e-6)).view(c, n, h, w).permute(1, 0, 2, 3)
        return x
    
    def _one_loss(self, fs, ft):
        if fs.shape[2:] != ft.shape[2:]:
            if self.resize_stu:
                fs = F.interpolate(fs, size=ft.shape[2:], mode='bilinear', align_corners=False)
            else:
                ft = F.interpolate(ft, size=fs.shape[2:], mode='bilinear', align_corners=False)
                
        if self.normalize:
            fs = self._feature_norm(fs)
            ft = self._feature_norm(ft)
            
        return self.mse(fs, ft) * 0.5
        
    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t), "Mismatch between student and teacher features"
        losses = [self._one_loss(s, t.detach()) for s, t in zip(y_s, y_t)]
        return torch.stack(losses).mean()


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller.lower()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.need_align = any(cs != ct for cs, ct in zip(channels_s, channels_t))
        
        if self.need_align:
            self.align_module = nn.ModuleList([
                nn.Conv2d(channel_s, channel_t, kernel_size=1, stride=1, padding=0).to(device)
                for channel_s, channel_t in zip(channels_s, channels_t)
            ])
            self.norm_t = nn.ModuleList([nn.BatchNorm2d(c, affine=False).to(device) for c in channels_t])
            self.norm_s = nn.ModuleList([nn.BatchNorm2d(c, affine=False).to(device) for c in channels_t])
        else:
            self.align_module = None
            self.norm_t = None
            self.norm_s = None
        
        aligned_channels_s = channels_t if self.need_align else channels_s
        self.feature_loss = self.build_feature_loss(distiller, channels_s, channels_t)
        
    def build_feature_loss(self, name, channels_s, channels_t):
        name = name.lower()
        if name == 'mgd':
            return MGDLoss(channels_s, channels_t)
        elif name == 'cwd':
            return CWDLoss()
        elif name == 'at':
            return ATLoss()
        elif name == 'skd':
            return SKDLoss()
        elif name == 'pkd':
            return PKDLoss(normalize=False)
        else:
            raise ValueError(f"Unknown distiller type: {name}")

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t), "Student and teacher feature length mismatch"
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if self.need_align and self.distiller != 'mgd':
                s = self.align_module[idx](s)
                s = self.norm_s[idx](s)  
                t = self.norm_t[idx](t)
                
            stu_feats.append(s)
            tea_feats.append(t.detach())

        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class Distillation_loss:
    def __init__(self, model_s, model_t, layers, distiller="CWDLoss", loss_weights=None):  # model must be de-paralleled
        self.distiller = distiller.lower()
        self.layers = list(map(int, layers))
        self.loss_weights = loss_weights or {'cwd': 0.15, 'mgd': 0.03, 'at': 0.033, 'skd': 0.25, 'pkd': 0.15}
        
        # distillation layers
        channels_s = self.get_channels(model_s, self.layers)
        channels_t = self.get_channels(model_t, self.layers)

        self.D_loss_fn = FeatureLoss(channels_s=channels_s, channels_t=channels_t, distiller=self.distiller)

        self.teacher_modules = [model_t.model[i] for i in self.layers]
        self.student_modules = [model_s.model[i] for i in self.layers]

        self.teacher_outputs = []
        self.student_outputs = []
        self.hooks = []
        
    def get_channels(self, model, layer_ids):
        channels = []
        for i in layer_ids:
            m = model.model[i]
            if hasattr(m, 'cv2') and hasattr(m.cv2, 'conv'):
                channels.append(m.cv2.conv.out_channels)
            else:
                raise AttributeError(f"Layer {i} does not have 'cv2.conv' attributes.")
        return channels

    def register_hooks(self):
        def make_hook(buffer):
            def hook_fn(module, input, output):
                buffer.append(output)
            return hook_fn

        for tm, sm in zip(self.teacher_modules, self.student_modules):
            self.hooks.append(tm.register_forward_hook(make_hook(self.teacher_outputs)))
            self.hooks.append(sm.register_forward_hook(make_hook(self.student_outputs)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_loss(self):
        assert len(self.teacher_outputs) == len(self.student_outputs), "Feature output mismatch"
        loss = self.D_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs)
        
        weight = self.loss_weights.get(self.distiller, 0.03)
        loss *= weight
            
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        return loss

