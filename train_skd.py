import argparse
import time
import datetime
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from models.sagan_models import Discriminator
from losses.loss import SegCrossEntropyLoss, CriterionKD, CriterionAdv, CriterionAdvForG, CriterionPairWise
from models.model_zoo import get_segmentation_model

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.datasets import ADETrainSet, ADEValSet, CSTrainValSet, VOCDataSet, VOCDataValSet, COCOTrainSet, COCOValSet
from utils.flops import cal_multi_adds, cal_param_size
from settings import *

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3', help='model name')  
    parser.add_argument('--student-model', type=str, default='deeplabv3', help='model name')                      
    parser.add_argument('--student-backbone', type=str, default='resnet18', help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101', help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys', choices=['citys', 'voc', 'ade20k', 'coco'], help='dataset name')
    parser.add_argument('--crop-size', type=str, default='512,512', help='crop image size: height,width')
    parser.add_argument('--workers', '-j', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1)
    
    # training hyper params
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument("--lr-d", type=float, default=4e-4, help="learning rate for D")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument("--lambda-kd", type=float, default=1.0, help="lambda_kd")
    parser.add_argument('--lambda-pa', default=10, type=float, help='lambda-pa')

    parser.add_argument("--lambda-adv", type=float, default=0.001, help="lambda_adv")
    parser.add_argument("--lambda-d", type=float, default=0.1, help="lambda_d")
    parser.add_argument("--adv-conv-dim", type=int, default=64, help="conv dim in adv")
    parser.add_argument("--preprocess-GAN-mode", type=int, default=1, help="preprocess-GAN-mode should be tanh or bn")
    
    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0') 
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)

    # checkpoint and log
    parser.add_argument('--save-dir', default='./ckpt', help='Directory for saving checkpoint models')
    parser.add_argument('--log-dir', default='./logs/', help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=50, help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=200000, help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=400, help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None', help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='./ckpt/deeplabv3_resnet101_citys_best_model.pth', help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='./pretrained/resnet18-imagenet.pth', help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None', help='pretrained seg model')
                        
    # evaluation only
    parser.add_argument('--skip-val', action='store_true', default=False, help='skip validation during training')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        h, w = map(int, args.crop_size.split(','))
        args.crop_size = (h, w)
        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(DATA_CS, list_path=DATALIST_CS_TRAIN, 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(DATA_CS, list_path=DATALIST_CS_VAL, 
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataSet(DATA_VOC, list_path=DATALIST_VOC_TRAIN, 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(DATA_VOC, list_path=DATALIST_VOC_VAL)
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(DATA_ADE20K, list_path=DATALIST_ADE20K_TRAIN, 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEValSet(DATA_ADE20K, list_path=DATALIST_ADE20K_VAL)
        elif args.dataset == 'coco':
            train_dataset = COCOTrainSet(DATA_COCO, list_path=DATALIST_COCO_TRAIN, 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = COCOValSet(DATA_COCO, list_path=DATALIST_COCO_VAL)
    
        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler,
                                            num_workers=args.workers, pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler,
                                          num_workers=args.workers, pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model, backbone=args.teacher_backbone,
                                            local_rank=args.local_rank, pretrained_base='None',
                                            pretrained=args.teacher_pretrained,
                                            norm_layer=nn.BatchNorm2d, aux=True,
                                            num_class=train_dataset.num_class).to(self.args.local_rank)

        self.s_model = get_segmentation_model(model=args.student_model, backbone=args.student_backbone,
                                            local_rank=args.local_rank, pretrained_base=args.student_pretrained_base,
                                            pretrained='None', aux=args.aux, norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)
        
        self.D_model = Discriminator(args.preprocess_GAN_mode, conv_dim=args.adv_conv_dim, input_channel=train_dataset.num_class, norm_layer=BatchNorm2d).to(self.device)
        
        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()
        self.D_model.eval()

        # create criterion
        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD().to(self.device)
        self.criterion_pairwise = CriterionPairWise().to(self.device)
        self.criterion_adv_for_G = CriterionAdvForG('hinge').to(self.device)

        self.criterion_adv = CriterionAdv('hinge').to(self.device)

        self.optimizer = torch.optim.SGD(self.s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.D_model.parameters(), args.lr_d, [0.9, 0.99])
        # self.optimizer_D = torch.optim.SGD(self.D_model.parameters(), lr=args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, device_ids=[args.local_rank], output_device=args.local_rank)
            self.D_model = nn.parallel.DistributedDataParallel(self.D_model, device_ids=[args.local_rank], output_device=args.local_rank)
            
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        return cur_lr
    
    def adjust_lr_D(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = cur_lr
        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(self.args.max_iterations))

        self.s_model.train()
        self.D_model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            
            images = images.to(self.device)
            targets = targets.long().to(self.device)
            
            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)
            
            if self.args.aux:
                task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
            else:
                task_loss = self.criterion(s_outputs[0], targets)
            
            kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], t_outputs[0])
            pairwise_loss = self.args.lambda_pa * self.criterion_pairwise(s_outputs[-1], t_outputs[-1])
            adv_G_loss = self.args.lambda_adv * self.criterion_adv_for_G(self.D_model(s_outputs[0]))

            losses = task_loss + kd_loss + pairwise_loss + adv_G_loss
            
            self.adjust_lr(base_lr=self.args.lr, iter=iteration-1, max_iter=self.args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            task_losses_reduced = self.reduce_mean_tensor(task_loss)
            kd_losses_reduced = self.reduce_mean_tensor(kd_loss)
            pairwise_losses_reduced = self.reduce_mean_tensor(pairwise_loss)
            adv_G_losses_reduced = self.reduce_mean_tensor(adv_G_loss)

            d_loss = self.args.lambda_d * self.criterion_adv(self.D_model(s_outputs[0].detach()), self.D_model(t_outputs[0].detach()))
            
            self.adjust_lr_D(base_lr=self.args.lr_d, iter=iteration-1, max_iter=self.args.max_iterations, power=0.9)
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            d_losses_reduced = self.reduce_mean_tensor(d_loss)

            eta_seconds = ((time.time() - start_time) / iteration) * (self.args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info("Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}" \
                    "|| PairWise Loss: {:.4f} || Adv_G Loss: {:.4f} || Adv_D Loss: {:.4f} " \
                    "|| Cost Time: {} || Estimated Time: {}".format(
                    iteration, self.args.max_iterations, self.optimizer.param_groups[0]['lr'], task_losses_reduced.item(),
                    kd_losses_reduced.item(), pairwise_losses_reduced.item(), 
                    adv_G_losses_reduced.item(), d_losses_reduced.item(),
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info("Total training time: {} ({:.4f}s / it)".format(total_training_str, total_training_time / self.args.max_iterations))
        logger.info("Best validation mIoU: {:.3f}".format(self.best_pred * 100))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            # logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
        
        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(self.args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(self.args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(self.args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(self.args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label) 
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc.item() * 100, mIoU * 100))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            logger.info("New best validation mIoU: {:.3f}".format(new_pred * 100))
        if (self.args.distributed is not True) or (self.args.distributed and self.args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best)
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'skd_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'skd_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), 
        filename='skd_{}_{}_{}_log.txt'.format(args.student_model, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
