import os
import jittor as jt
import jittor.nn as nn
import numpy as np
from datetime import datetime
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
from lib.Network_Res2Net_GRA_NCD import Network


jt.flags.use_cuda = 1
jt.flags.log_silent = True


def structure_loss(pred, mask):
    weit = 1 + 5 * jt.abs(nn.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = nn.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit * wbce).sum(2, 3) / weit.sum(2, 3)

    pred = jt.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(2, 3)
    union = ((pred + mask) * weit).sum(2, 3)
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.float32()
            gts = gts.float32()

            preds = model(images)
            loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts)
            loss_final = structure_loss(preds[3], gts)
            loss = loss_init + loss_final

            optimizer.backward(loss)

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(), loss_init.item(),loss_final.item()))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.item(), loss_init.item(), loss_final.item()))

                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.item(), 'Loss_final': loss_final.item(),
                                    'Loss_total': loss.item()},
                                   global_step=step)


                rgb_img = images[0].numpy().transpose(1, 2, 0)  # CHW -> HWC
                writer.add_image('RGB', rgb_img, step, dataformats='HWC')
                gt_img = gts[0].numpy().squeeze()
                writer.add_image('GT', gt_img, step, dataformats='HW')

                res = preds[0][0].sigmoid().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', res, step, dataformats='HW')
                res = preds[3][0].sigmoid().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', res, step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            jt.save(model.state_dict(), save_path + 'Net_epoch_{}.pkl'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        jt.save(model.state_dict(), save_path + 'Net_epoch_{}.pkl'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    global best_mae, best_epoch
    model.eval()
    with jt.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, _ = test_loader[i]

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            # 将 NumPy 数组转换为 Jittor 张量
            image = jt.array(image)  # 添加这行转换

            # 3. 修正维度问题 - 添加batch维度
            if image.ndim == 3:  # 如果是3维 (C, H, W)
                image = image.unsqueeze(0)  # 添加batch维度 -> (1, C, H, W)

            # 4. 确保数据类型正确
            image = image.float32()

            res = model(image)

            res = nn.interpolate(res[3], size=gt.shape, mode='bilinear', align_corners=False)
            # res = res.sigmoid().squeeze().numpy()
            res_sig = res.sigmoid()
            res_squeezed = res_sig.squeeze()
            res = res_squeezed.numpy()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', mae, global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                jt.save(model.state_dict(), save_path + 'Net_epoch_best.pkl')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=18, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./Dataset/TrainValDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot_jittor_big_30/SINet_V2/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    print(f'USE GPU {opt.gpu_id}')



    model = Network(channel=32)

    if opt.load is not None:
        model.load_state_dict(jt.load(opt.load))
        print('load model from ', opt.load)

    optimizer = nn.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 在参数解析后添加
    if jt.in_mpi:
        jt.world_size = jt.mpi.world_size()
        jt.rank = jt.mpi.rank()
    else:
        jt.world_size = 1
        jt.rank = 0

    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)

    if jt.world_size > 1:
        train_loader = train_loader.set_attrs(
            num_workers=8,
            batch_size=opt.batchsize // jt.world_size,
            shuffle=True,
            drop_last=True
        )

    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)