import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./model')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from model.DSCDNet import DSCDNet
from data import get_loader,test_dataset
from utils import clip_gradient
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss.ssim import SSIM, FocalLoss
#from loss.loss import SLSIoULoss
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

#build the model
model = DSCDNet()
# if(opt.load is not None):
#     model.load_pre(opt.load)
#     print('load model from ',opt.load)
# model = nn.DataParallel(model).cuda()
model = model.cuda()


base, body = [], []
for name, param in model.named_parameters(): 
        # if 'backbone' in name or 'swin_thermal' in name:
        if 'backbone' in name :
            print('base')
            print(name)
            base.append(param)
        else:
            print('body')
            print(name)
            body.append(param)
optimizer = torch.optim.SGD([{'params': base,'lr': opt.lr}, {'params': body,'lr': opt.lr}],  momentum=opt.momentum,
                                weight_decay=opt.decay_rate, nesterov=True)


#set the path
train_root = opt.train_data_root
test_root = opt.val_data_root

save_path=opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
num_gpus = torch.cuda.device_count()
print(f"========>num_gpus:{num_gpus}==========")
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+datetime.now().strftime("%m.%d") +'.withoutCAFA.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("SwinMCNet-Train")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path))


w_focal = 1.0      
w_iou = 1.0        
w_ssim = 0.5       

ds_weights = [0.2, 0.2, 0.3, 0.3, 0.5, 1.0]
# loss
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
ssim_loss = SSIM(window_size=12, size_average=True)
focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

step=0
writer = SummaryWriter(save_path+'summary')
best_mae=1
best_epoch=1
loss_kl = nn.KLDivLoss(reduction='batchmean')
#train function
def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    #SLSLOSS = SLSIoULoss()
    model_num = 4
    try:
        for i, (images, ts, gts, bodys, details) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            image, t, gt, body, detail = images.cuda(), ts.cuda(), gts.cuda(), bodys.cuda(), details.cuda()

            outb1, outd1, out1, outb2, outd2, out2  = model(image,t)

            loss_b1 = w_focal * focal_loss(outb1, gt) + w_iou * iou_loss(outb1, gt)
            loss_d1 = w_focal * focal_loss(outd1, gt) + w_iou * iou_loss(outd1, gt)
            loss_1 = w_focal * focal_loss(out1, gt) + w_iou * iou_loss(out1, gt)
            loss_b2 = w_focal * focal_loss(outb2, gt) + w_iou * iou_loss(outb2, gt)
            loss_d2 = w_focal * focal_loss(outd2, gt) + w_iou * iou_loss(outd2, gt)
            loss_2 = w_focal * focal_loss(out2, gt) + w_iou * iou_loss(out2, gt) + w_ssim * ssim_loss(out2, gt)

            loss = (ds_weights[0] * loss_b1 + 
                    ds_weights[1] * loss_d1 + 
                    ds_weights[2] * loss_1 + 
                    ds_weights[3] * loss_b2 + 
                    ds_weights[4] * loss_d2 + 
                    ds_weights[5] * loss_2)

            # loss2 = F.binary_cross_entropy_with_logits(out2, gt) + iou_loss(out2, gt) + ssim_loss(out2, gt)
            # loss = loss2
            
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            step += 1
            epoch_step += 1
            loss_all += loss.data
            
            if i % 50 == 0 or i == total_step or i == 1:
                print('%s | epoch:%d/%d | step:%d/%d | lr=%.6f | loss=%.6f '
                      % (datetime.now(), epoch, opt.epoch, i, total_step, 
                         optimizer.param_groups[1]['lr'], loss.item()))
                
                logging.info('##TRAIN##:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], lr_bk: {:.6f}, Loss: {:.4f} '
                            .format(epoch, opt.epoch, i, total_step, 
                                    optimizer.param_groups[1]['lr'], loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = out2[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out2', torch.tensor(res), step, dataformats='HW')
        
        loss_all /= epoch_step
        logging.info('##TRAIN##:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path+'CDFF_epoch_{}_withoutCAFA.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'CDFF_epoch_{}_withoutCAFA.pth'.format(epoch))
        print('save checkpoints successfully!')
        
#test function
@torch.no_grad()
def test(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        #for i in range(1000):
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            t = t.cuda()
            #shape = (W,H)
            outb1, outd1, out1, outb2, outd2, out2  = model(image,t)
            res = out2
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae=mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('\n')
        print('##TEST##:Epoch: {}   MAE: {}'.format(epoch,mae))
        
        if epoch==1:
            best_mae=mae
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model.state_dict(), save_path+'CDFF_epoch_best_withoutCAFA.pth')
        print('##SAVE##:bestEpoch: {}   bestMAE: {}'.format(best_epoch,best_mae))
        print('\n')
        logging.info('##TEST##:Epoch:{}   MAE:{}   bestEpoch:{}   bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
# if __name__ == '__main__':
#     print("Start train...")
#     decay_rate = 0.6  # 学习率衰减比例
#     decay_epoch = 10  # 每隔多少个epoch衰减一次
#     for epoch in range(1, opt.epoch + 1):
#         if epoch % decay_epoch == 0 and epoch != 1:  # 防止在第一个epoch时衰减
#              optimizer.param_groups[0]['lr'] *= decay_rate
#         # optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
#         # optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
#              writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
#
#         # print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
#         train(train_loader, model, optimizer, epoch,save_path)
#         test(test_loader,model,epoch,save_path)

if __name__ == '__main__':
    print("Start train...")
    decay_rate = 0.6  # 学习率衰减比例
    decay_epoch = 10  # 每隔多少个epoch衰减一次
    for epoch in range(1, opt.epoch + 1):
        # if 10 <= epoch <= 80 and epoch % 10 == 0:  # epoch处于20-80时衰减
        #     optimizer.param_groups[0]['lr'] *= decay_rate
        #     optimizer.param_groups[1]['lr'] *= decay_rate
        #     writer.add_scalar('lr', optimizer.param_groups[1]['lr'], global_step=epoch)
        #
        #     print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[1]['lr']}")
        if epoch < 60:
            # lr = (1 - abs((epoch) / 60 * 2 - 1)) * opt.lr
            optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / 60 * 2 - 1)) * opt.lr * 0.1
            optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / 60 * 2 - 1)) * opt.lr
        else:
            # lr = opt.lr * (1 - abs((59) / 60 * 2 - 1))
            optimizer.param_groups[0]['lr'] = opt.lr * (1 - abs((59) / 60 * 2 - 1)) * 0.1
            optimizer.param_groups[1]['lr'] = opt.lr * (1 - abs((59) / 60 * 2 - 1))
        # optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
        # optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        train(train_loader, model, optimizer, epoch,save_path)
        test(test_loader,model,epoch,save_path)