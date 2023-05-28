"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable

__all__ = ['SegCrossEntropyLoss', 'CriterionKD', 'CriterionCWD', 'CriterionMGD',
            'CriterionDIST', 'CriterionMiniBatchCrossImagePair', 'CriterionIFV', 'CriterionPairWise', 
           'CriterionAdv', 'CriterionAdvForG', 'CriterionAdditionalGP']

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

# TODO: optim function
class SegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(SegCrossEntropyLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        B, H, W = targets.size()
        inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
        return self.task_loss(inputs, targets)


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1):
        super(CriterionKD, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft):
        B, C, h, w = soft.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        soft = soft.permute(0, 2, 3, 1).contiguous().view(-1, C)
        p_s = F.log_softmax(pred / self.temperature, dim=1)
        p_t = F.softmax(soft / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        return loss


class CriterionCWD(nn.Module):
    def __init__(self, temperature=4.0, BatchNorm2d=nn.SyncBatchNorm):
        super(CriterionCWD, self).__init__()
        self.temperature = temperature
        self.linear = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), BatchNorm2d(256), torch.nn.ReLU(True))

    def forward(self, preds_S, preds_T, use_linear=True):
        if use_linear:
            preds_S = self.linear(preds_S)
        n, c, h, w = preds_S.shape
        preds_S = preds_S.reshape((n, c, -1))
        preds_S = F.log_softmax(preds_S / self.temperature, dim=-1)
        with torch.no_grad():
            preds_T = preds_T.reshape((n, c, -1))
            preds_T = F.softmax(preds_T / self.temperature, dim=-1)

        loss = F.kl_div(preds_S, preds_T.detach(), reduction='sum') * (self.temperature**2)
        loss /= n * c
        return loss


class CriterionMGD(nn.Module):
    def __init__(self, lambda_mgd):
        super(CriterionMGD, self).__init__()
        self.lambda_mgd = lambda_mgd
        self.align = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.generation = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        preds_S = self.align(preds_S)
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        loss_mse = nn.MSELoss(reduction='sum')
        loss = loss_mse(new_fea, preds_T) / N

        return loss


class CriterionDIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0):
        super(CriterionDIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def cosine_similarity(self, a, b, eps=1e-8):
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

    def pearson_correlation(self, a, b, eps=1e-8):
        return self.cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)
    
    def inter_class_relation(self, y_s, y_t):
        return 1 - self.pearson_correlation(y_s, y_t).mean()
    
    def intra_class_relation(self, y_s, y_t):
        return self.inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    def forward(self, preds_S, preds_T):
        B, C, h, w = preds_S.size()
        preds_S = preds_S.permute(0, 2, 3, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # preds_S = F.softmax(preds_S, dim=-1)
        # preds_T = F.softmax(preds_T, dim=-1)
        inter_loss = self.inter_class_relation(preds_S, preds_T)
        intra_loss = self.intra_class_relation(preds_S, preds_T)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        
        return loss


class CriterionMiniBatchCrossImagePair(nn.Module):
    def __init__(self, temperature):
        super(CriterionMiniBatchCrossImagePair, self).__init__()
        self.temperature = temperature

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output


    def forward(self, feat_S, feat_T):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        '''
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        '''
        
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        device = feat_S.device
        sim_dis = torch.tensor(0.).to(device)
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_
        sim_dis = sim_dis / (B * B)
        return sim_dis


class CriterionIFV(nn.Module):
    def __init__(self, classes=19):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes

    def forward(self, preds_S, preds_T, target):
        feat_S = preds_S # x_feat_after_psp, (B, C1, H1, W1)
        feat_T = preds_T
        feat_T.detach()
        device = feat_S.device
        size_f = (feat_S.shape[2], feat_S.shape[3]) # (H1, W1)
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size()) # (B, 1, H, W) -> (B, 1, H1, W1) -> (B, C1, H1, W1)
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())

        center_feat_S = feat_S.clone() # (B, C1, H1, W1)
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            #               NON-TARGET, KEEP                    TARGET, CHANGE, (B, C1, H1, W1) * (AVE(B, C1), 1, 1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S) # (B, H1, W1)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return loss


class CriterionPairWise(nn.Module):
    def __init__(self):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWise, self).__init__()

    def similarity(self, feat):
        feat = F.normalize(feat, p=2, dim=1)
        feat = feat.reshape(feat.shape[0], feat.shape[1], -1) # (B, C, 2*2)
        return torch.einsum('icm,icn->imn', [feat, feat]) # (B, 4, 4)

    def sim_dis_compute(self, f_S, f_T): # (B, C, 2, 2)
        sim_err = ((self.similarity(f_T) - self.similarity(f_S))**2) / ((f_T.shape[-1] * f_T.shape[-2])**2) / f_T.shape[0]
        sim_dis = sim_err.sum()
        return sim_dis

    def forward(self, feat_S, feat_T):
        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = 2, 2
        maxpool = nn.MaxPool2d(kernel_size = (patch_w, patch_h), stride = (patch_w, patch_h), padding = 0, ceil_mode = True)
        loss = self.sim_dis_compute(maxpool(feat_S), maxpool(feat_T))
        return loss


class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S):
        g_out_fake = d_out_S
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake


class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S.shape == d_out_T.shape, 'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake


class CriterionAdditionalGP(nn.Module):
    def __init__(self, D_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.D = D_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_S, d_in_T):
        assert d_in_S.shape == d_in_T.shape, 'the output dim of D with teacher and student as input differ'

        real_images = d_in_T
        fake_images = d_in_S

        # Compute gradient penalty
        device = real_images.device
        alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device).expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0], inputs=interpolated, grad_outputs=torch.ones(out[0].size()).to(device),
                                    retain_graph=True, create_graph=True, only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss