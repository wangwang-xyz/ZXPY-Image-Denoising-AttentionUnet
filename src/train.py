import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mydataset import MyData
from network import ModefiedUnet
import skimage.metrics

def score(result_write_data, gt, white_level):
    psnr = skimage.metrics.peak_signal_noise_ratio(
        gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
    ssim = skimage.metrics.structural_similarity(
        gt.astype(np.float), result_write_data.astype(np.float), multichannel=True, data_range=white_level)

    return psnr, ssim

def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)

    return output_data

def write_image(input_data, height, width):
    height = height * 2
    width = width * 2
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]

    return output_data

def train(args):
    address = args.root_dir
    dataname = args.data_path
    gtname = args.ground_path
    train_test_ratio = args.train_test_ratio
    batch_size = args.batch_size
    learning_rate = args.lr
    epoch = args.epoch
    black_level = args.black_level
    white_level = args.white_level

    # 设置训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    data_idx = np.array([x for x in range(0, 100)])
    np.random.shuffle(data_idx)
    train_idx = data_idx[0:int(train_test_ratio*100)]
    test_idx = data_idx[int(train_test_ratio*100):]
    train_set = MyData(address, dataname, gtname, train_idx)
    test_set = MyData(address, dataname, gtname, test_idx)

    train_set_size = len(train_set)
    test_set_size = len(test_set)
    print("Size of Train set： {}".format(train_set_size))
    print("Size of Test set： {}".format(test_set_size))


    # 数据加载
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # 加载网络模型
    model = ModefiedUnet()
    model = model.to(device)
    if args.pretrained:
        pretrained_model = torch.load(args.pretrained_model, map_location=device)
        pretrained_model_dict = pretrained_model#.state_dict()
        model_dict = model.state_dict()
        model_dict.update(pretrained_model_dict)
        # pretrained_dict = {k: v for k, v in pretrained_model_dict if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(model_dict)
        print("Pretrained Model has been loaded")

    # 定义损失函数
    loss_fn = nn.MSELoss(reduction='sum')
    loss_fn = loss_fn.to(device)

    # 定义优化器
    lf = learning_rate
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lf)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 2400], gamma = 0.5)

    # 训练
    train_step = 0
    # 添加 TensorBoard
    writer = SummaryWriter("logs/train")

    best_model = model
    min_loss = 1000000
    for i in range(epoch):
        print("------------------Epoch({}/{})------------------".format(i+1, epoch))

        model.train()
        for data in train_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = train_step + 1
            if train_step % 20 == 0:
                print("Training times: {}，Loss: {}".format(train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), train_step)

        # 调整学习率
        scheduler.step()

        # 测试
        average_psnr = 0
        average_ssmi = 0
        total_test_loss = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                [b, c, height, width] = targets.shape
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets)

                result_data = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)
                result_data = inv_normalization(result_data, black_level, white_level)
                result_write_data = write_image(result_data, height, width)
                result_gt = targets.cpu().detach().numpy().transpose(0, 2, 3, 1)
                result_gt = inv_normalization(result_gt, black_level, white_level)
                result_write_gt = write_image(result_gt, height, width)

                total_test_loss = total_test_loss + loss
                psnr, ssmi = score(result_write_data, result_write_gt, white_level)
                average_ssmi = average_ssmi + ssmi
                average_psnr = average_psnr + psnr

        average_psnr = average_psnr / test_set_size
        average_ssmi = average_ssmi / test_set_size
        print("Average PSNR: {}".format(average_psnr))
        print("Average SSMI: {}".format(average_ssmi))
        print("Loss on test set: {}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss.item()/test_set_size, i+1)

        if total_test_loss < min_loss:
            min_loss = total_test_loss
            best_model = model

    print("------------------Best Model------------------")
    print("Loss of best model of test set: {}".format(min_loss))
    torch.save(best_model, "./model/model.pth")
    print("Best Model Saved")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1E-3)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model', type=str, default=r"./lib/baseline/baseline/models/th_model.pth")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--root_dir', type=str, default=r"../dataset/dataset/")
    parser.add_argument('--data_path', type=str, default=r"noisy/")
    parser.add_argument('--ground_path', type=str, default=r"ground truth/")
    parser.add_argument('--train_test_ratio', type=float, default=0.8)

    args = parser.parse_args()
    print(args)
    train(args)
