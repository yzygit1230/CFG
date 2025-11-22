import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataloader import BreastSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
from networks.Swin_MAE import swin_mae
from networks.unet_model import UNet
from networks.kan_unet import KAN_UNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='breast')
parser.add_argument("--save_name", type=str, default="debug")
parser.add_argument("--overwrite", default='True')
parser.add_argument("--model", type=str, default="kan")
parser.add_argument("--swin_chkpt_dir", type=str, default="code/networks/Swin_MAE/Swin_MAE.pth")
parser.add_argument("--data_dir", type=str, default="data/BreastSlice")
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.03)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--lamda", type=float, default=0.95)
parser.add_argument('--amp', type=int, default=1)
parser.add_argument("--label_bs", type=int, default=4)
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=4)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=324)
parser.add_argument("--ema_decay", type=float, default=0.99)
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument("--IDG_prob", default=1.0, type=float)
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, model_tea, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(model_tea.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)       

part = ['base'] 
dataset = BreastSegmentation
n_part = len(part)
dice_calcu = {'breast':metrics.dice_coeff}

swin_chkpt_dir = args.swin_chkpt_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mae = getattr(swin_mae, 'swin_mae')().to(device)
checkpoint = torch.load(swin_chkpt_dir, map_location='cpu')
model_mae.load_state_dict(checkpoint['model'], strict=True)

def obtain_biIDG_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break
    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_mae_image(images):
    images = images.repeat(1, 3, 1, 1)
    images = images.float().cuda()
    _, y, mask_mae = model_mae(images)
    y = model_mae.unpatchify(y)
    y = y.detach().cpu()
    mask_mae = mask_mae.detach()
    mask_mae = mask_mae.unsqueeze(-1).repeat(1, 1, model_mae.patch_embed.patch_size ** 2 * 3) 
    mask_mae = model_mae.unpatchify(mask_mae)
    mask_mae = mask_mae.detach().cpu()
    images = images.detach().cpu()
    y = y * mask_mae
    im_paste = images * (1 - mask_mae) + y * mask_mae
    im_paste = im_paste.mean(dim=1, keepdim=True)
    
    return im_paste

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    val_dice = [0.0] * n_part
    domain_num = len(test_dataloader)
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        for batch_num, sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            mask = mask.eq(0).long()
            output = model(data)
            loss_seg = ce_loss(output, mask).mean() + \
                        dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
            dice = dice_calcu[args.dataset](np.asarray(torch.max(torch.softmax(output.cpu(),dim=1), dim=1)[1]),mask.clone().cpu())
            
            domain_val_loss += loss_seg.item()
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]

        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, dc, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (dc, epoch, domain_val_loss)
        for n, p in enumerate(part):
            text += ' val_%s_dice: %f' % (p, domain_val_dice[n])
            if n != n_part-1:
                text += ','
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    for n, p in enumerate(part):
        text += ' val_%s_dice: %f' % (p, val_dice[n])
        if n != n_part-1:
            text += ','
    logging.info(text)
    return val_dice

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    num_channels = 1
    patch_size = 224
    num_classes = 2
    max_iterations = args.max_iterations
    trans = transforms.Compose([tr.RandomScaleCrop(patch_size)])
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    domain = list(range(1, domain_num+1))
    domain_len = [648, 452, 114, 224]
    lb_domain = args.lb_domain

    lb_num = args.lb_num
    lb_idxs = list(range(lb_num)) 
    data_num = domain_len[lb_domain-1] 
    unlabeled_idxs = list(range(lb_num, data_num)) 
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=args.data_dir, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, transform=trans, normal_toTensor=normal_toTensor)
    lb_dataloader = DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    ulb_dataset = dataset(base_dir=args.data_dir, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, transform=trans, normal_toTensor=normal_toTensor)
    ulb_dataloader = DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    for i in range(1, domain_num + 1):
        cur_dataset = dataset(base_dir=args.data_dir, phase='val', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)
    for i in range(0, domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=2, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(name='kan', ema=False):
        if name == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes)
        if name == 'kan':
            model = KAN_UNet()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model(name = args.model)
    model_tea = create_model(name = args.model, ema=True)

    iter_num = 0
    start_epoch = 0

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes)

    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)
    lamda = args.lamda

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        model_tea.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')

        for i_batch in range(1, args.num_eval_iter+1):
            lb_dataloader_iter = iter(lb_dataloader)
            lb_sample = next(lb_dataloader_iter)
            ulb_dataloader_iter = iter(ulb_dataloader)
            ulb_sample = next(ulb_dataloader_iter)
            
            lx, lx_label = lb_sample['image'], lb_sample['label']
            ulx = ulb_sample['image']
            lx, lx_label, ulx = lx.cuda(), lx_label.cuda(), ulx.cuda()
            lb_mask = lx_label.eq(0).long()

            with amp_cm():
                with torch.no_grad():
                    label_box = torch.stack([obtain_biIDG_box(img_size=patch_size, p=args.IDG_prob) for i in range(len(ulx))], dim=0)
                    img_box = label_box.unsqueeze(1)  
                    logits_ulx = model_tea(ulx) 
                    
                    prob_ulx = torch.softmax(logits_ulx, dim=1) 
                    prob, pseudo_label = torch.max(prob_ulx, dim=1)
                    mask = (prob > lamda).unsqueeze(1).float()     

                mask_in, mask_out = mask.clone(), mask.clone()
                mask_in[img_box.expand(mask_in.shape) == 1] = 1     
                mask_out[img_box.expand(mask_out.shape) == 0] = 1     
                pseudo_label_in = (pseudo_label * (1-label_box) + lb_mask * label_box).long()   
                pseudo_label_out = (lb_mask * (1-label_box) + pseudo_label * label_box).long()   

                lx_mae = obtain_mae_image(lx).to(device)
                u_in = ulx * (1-img_box) + lx * img_box 
                u_out = lx * (1-img_box) + ulx * img_box   

                logits_lx = model(lx)  
                logits_lx_mae = model(lx_mae) 
                logits_u_in = model(u_in)   
                logits_u_out = model(u_out)

                u_in_mae = ulx * (1-img_box) + lx_mae * img_box   
                u_out_mae = lx_mae * (1-img_box) + ulx * img_box  

                logits_u_in_mae = model(u_in_mae)   
                logits_u_out_mae = model(u_out_mae)

                sup_loss = ce_loss(logits_lx, lb_mask).mean() + \
                            dice_loss(logits_lx, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                sup_loss_mae = ce_loss(logits_lx_mae, lb_mask).mean() + \
                            dice_loss(logits_lx_mae, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
        
                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))
                unsup_loss_in = (ce_loss(logits_u_in, pseudo_label_in) * mask_in.squeeze(1)).mean() + \
                                dice_loss(logits_u_in, pseudo_label_in.unsqueeze(1), mask=mask_in, softmax=softmax, sigmoid=sigmoid, multi=multi)
                unsup_loss_out = (ce_loss(logits_u_out, pseudo_label_out) * mask_out.squeeze(1)).mean() + \
                                dice_loss(logits_u_out, pseudo_label_out.unsqueeze(1), mask=mask_out, softmax=softmax, sigmoid=sigmoid, multi=multi)
                unsup_loss_in_mae = (ce_loss(logits_u_in_mae, pseudo_label_in) * mask_in.squeeze(1)).mean() + \
                                dice_loss(logits_u_in_mae, pseudo_label_in.unsqueeze(1), mask=mask_in, softmax=softmax, sigmoid=sigmoid, multi=multi)
                unsup_loss_out_mae = (ce_loss(logits_u_out_mae, pseudo_label_out) * mask_out.squeeze(1)).mean() + \
                                dice_loss(logits_u_out_mae, pseudo_label_out.unsqueeze(1), mask=mask_out, softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                loss = sup_loss + sup_loss_mae + consistency_weight * (unsup_loss_in + unsup_loss_out + unsup_loss_in_mae + unsup_loss_out_mae) 

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            update_ema_variables(model, model_tea, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('train/mask', mask.mean(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/unsup_loss_in', unsup_loss_in.item(), iter_num)
            writer.add_scalar('train/unsup_loss_out', unsup_loss_out.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/bi_consistency_weight', consistency_weight**2, iter_num)
            if p_bar is not None:
                p_bar.update()

            p_bar.set_description('iteration %d : loss:%.3f, sup_loss:%.3f, unsup_loss_in:%.3f, unsup_loss_out:%.3f, cons_w:%.3f, mask_ratio:%.3f' 
                                % (iter_num, loss.item(), sup_loss.item(), unsup_loss_in.item(), unsup_loss_out.item(), consistency_weight, mask.mean()))

        if p_bar is not None:
            p_bar.close()

        logging.info('test stu model')
        stu_val_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model_{}.pth".format(args.model,epoch_num)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            torch.save(model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
        text = 'checkpoint.pth'
        checkpoint_path = os.path.join(snapshot_path, text)
        util.save_osmancheckpoint(epoch_num+1, model_tea, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter, checkpoint_path)
        logging.info('save checkpoint to {}'.format(checkpoint_path))

    writer.close()


if __name__ == "__main__":
    snapshot_path = args.dataset + "-test/" + args.save_name + "/"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print(snapshot_path)
    if os.path.exists(snapshot_path):
        shutil.rmtree(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)