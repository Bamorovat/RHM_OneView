import os
import time
import numpy as np
import torch
import glob
from configs.config_F import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset_preparation import VideoDataset
from models import C3D_model, R2Plus1D_model, R3D_model, Slow_Fast_model
from tensorboardX import SummaryWriter
import neptune.new as neptune
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

run = neptune.init(
    name="one_C3D",
    # source_files=["train_new.py", "config_O.py", "dataset_preparation.py", "models/C3D_model.py"],
    description='Model:C3D, View: Front, situation:Normall',
    project="m.bamorovat/HAR-One-newframes",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMzk0NzVhZi05MmJmLTQzZWEtYjU2Ni01YjUwMzA4MmZkN2MifQ==",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device being used:", device)

y_pred = []
y_true = []
best_test_acc1 = 0.0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    print('------------------ Start Training --------------------------')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, 1, len(train_dataloader))
    print(print_string)

    train_loop = tqdm(enumerate(train_dataloader), total = len(train_dataloader), leave=False)

    #for step, (inputs1, labels) in enumerate(train_dataloader):
    for step, (inputs1,labels) in train_loop:
        data_time.update(time.time() - end)

        inputs1 = inputs1.to(device)
        labels = labels.to(device)
        outputs = model(inputs1)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss

        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs1.size(0))
        top1.update(prec1.item(), inputs1.size(0))
        top5.update(prec5.item(), inputs1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        train_loop.set_description(f"Epoch [{epoch}/{params['epoch_num']}]")
        train_loop.set_postfix(loss=losses.avg, top1=top1.avg, top5=top5.avg)

        #if (step + 1) % params['display'] == 0:
    print('---------------------- Train ---------------------------------')
            # for param in optimizer.param_groups:
            #     print('lr: ', param['lr'])
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
    print(print_string)
    print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val, batch_time=batch_time.val)
    print(print_string)
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg, top5_acc=top5.avg)
    print(print_string)

    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)

    run["train_loss_epoch"].log(losses.avg, epoch)
    run["train_top1_acc_epoch"].log(top1.avg, epoch)
    run["train_top5_acc_epoch"].log(top5.avg, epoch)


def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()

    print('---- Start Validation ----')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, 1, len(val_dataloader))
    print(print_string)

    with torch.no_grad():
        val_loop = tqdm(enumerate(val_dataloader), total = len(val_dataloader), leave=False)
        #for step, (inputs1, labels) in enumerate(val_dataloader):
        for step, (inputs1, labels) in val_loop:
            data_time.update(time.time() - end)
            inputs1 = inputs1.to(device)
            labels = labels.to(device)
            outputs = model(inputs1)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs1.size(0))
            top1.update(prec1.item(), inputs1.size(0))
            top5.update(prec5.item(), inputs1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            #if (step + 1) % params['display'] == 0:
            val_loop.set_description(f"Epoch [{epoch}/{params['epoch_num']}]")
            val_loop.set_postfix(loss=losses.avg, top1=top1.avg, top5=top5.avg)

    print('---- Validation ----')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
    print(print_string)
    print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val, batch_time=batch_time.val)
    print(print_string)
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,top5_acc=top5.avg)
    print(print_string)

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)

    run["val_loss_epoch"].log(losses.avg, epoch)
    run["val_top1_acc_epoch"].log(top1.avg, epoch)
    run["val_top5_acc_epoch"].log(top5.avg, epoch)

    return top1.avg


def test(model, test_dataloader, epoch, criterion, optimizer, writer, better):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()

    print('---- Start Test ----')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch,  1, len(test_dataloader))
    print(print_string)

    with torch.no_grad():
        test_loop = tqdm(enumerate(test_dataloader), total = len(test_dataloader), leave=False)
        #for step, (inputs1, labels) in enumerate(test_dataloader):
        for step, (inputs1, labels) in test_loop:
            data_time.update(time.time() - end)
            inputs1 = inputs1.to(device)
            labels = labels.to(device)
            outputs = model(inputs1)
            loss = criterion(outputs, labels)

            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction
            labels_1 = labels.data.cpu().numpy()
            y_true.extend(labels_1)  # Save Truth

            # measure accuracy and record loss

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs1.size(0))
            top1.update(prec1.item(), inputs1.size(0))
            top5.update(prec5.item(), inputs1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            test_loop.set_description(f"Epoch [{epoch}/{params['epoch_num']}]")
            test_loop.set_postfix(loss=losses.avg, top1=top1.avg, top5=top5.avg)

    top1_acc = top1.avg
           # if (step + 1) % params['display'] == 0:
    print('---- End Test ----')
    print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(test_dataloader))
    print(print_string)
    print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val, batch_time=batch_time.val)
    print(print_string)
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg, top5_acc=top5.avg)
    print(print_string)

    writer.add_scalar('test_loss_epoch', losses.avg, epoch)
    writer.add_scalar('test_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('test_top5_acc_epoch', top5.avg, epoch)

    run["test_loss_epoch"].log(losses.avg, epoch)
    run["test_top1_acc_epoch"].log(top1.avg, epoch)
    run["test_top5_acc_epoch"].log(top5.avg, epoch)
    
    global best_test_acc1
    # best = max(top1_acc, best_test_acc1)
    if top1_acc > best_test_acc1:
    # if better:
        # constant for classes
        classes = ['Bending', 'SittingDown', 'ClosingCan', 'Reaching', 'Walking', 'Drinking', 'StairsClimbingUp',
               'StairsClimbingDown', 'StandingUp', 'OpeningCan', 'CarryingObject', 'Cleaning', 'PuttingDownObjects',
               'LiftingObject']

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
        plt.figure(figsize=(15, 10))
        sn.heatmap(df_cm, annot=True)

        save_name = '/home/mb19aag/test/RH_HAR_Oneview/output' + 'confM' + '-' + params['view1'] + params['situation'] + '.' + 'png'
        plt.savefig(save_name)
        
        run['model/confusion'] = neptune.types.File(save_name)
        run["Best_test_loss_epoch"].log(losses.avg, epoch)
        run["Best_test_top1_acc_epoch"].log(top1.avg, epoch)
        run["Best_test_top5_acc_epoch"].log(top5.avg, epoch)
        
    best_test_acc1 = max(top1_acc, best_test_acc1)

def main():
    val_acc = 0.0
    cudnn.benchmark = False
    useTest = params['useTest']
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    saveName = params['model_name'] + '_' + params['view1'] + '_' + str(params['weight_decay'])
    #saveName = 'testgood'
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))    
    logdir = os.path.join('log', cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")

    train_dataloader = \
        DataLoader(
            VideoDataset(view1=params['view1'], situation= params['situation'], split='train', clip_len=params['clip_len']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            VideoDataset(view1=params['view1'], situation= params['situation'], split='val', clip_len=params['clip_len']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    test_dataloader = \
        DataLoader(
            VideoDataset(view1=params['view1'], situation= params['situation'], split='test', clip_len=params['clip_len']),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    print("load model")
    modelName = params['model_name']
    
    if modelName == 'C3D_model':
        model = C3D_model.C3D(num_classes=params['num_classes'], pretrained=params['pretrained'])
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': params['learning_rate']},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': params['learning_rate'] * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=params['num_classes'], layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': params['learning_rate']},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': params['learning_rate'] * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=params['num_classes'], layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif modelName == 'Slow_Fast':
        model = Slow_Fast_model.resnet50(class_num=params['num_classes'])
        train_params = model.parameters()    
    else:
        print('The model is not defined.')
        raise NotImplementedError

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(train_params, lr=params['learning_rate'], momentum=params['momentum'],
                          weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    run["parameters"] = params

    best_acc1 = 0.0
    for epoch in range(params['epoch_num']):
        for phase in ['train', 'val']:
            if phase == 'train':
                train(model, train_dataloader, epoch, criterion, optimizer, writer)
            elif phase == 'val':
                val_acc = validation(model, val_dataloader, epoch, criterion, optimizer, writer)
            scheduler.step()

            acc1 = val_acc
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            # if epoch % save_epoch == (save_epoch - 1):
        better = False
        # if is_best:
            # better = True
            #torch.save({
            #    'epoch': epoch + 1,
            #    'best_acc1': best_acc1,
            #    'state_dict': model.state_dict(),
            #    'opt_dict': optimizer.state_dict(),
            #}, os.path.join(save_dir, 'models', saveName + '.pth.tar'))
            #print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '.pth.tar')))
            # print("Saved Epoch Number is: ", epoch)

            #test(model, test_dataloader, epoch, criterion, optimizer, writer, better)

        #if useTest and epoch % params['nTestInterval'] == (params['nTestInterval'] - 1):
        test(model, test_dataloader, epoch, criterion, optimizer, writer, better)

    writer.close
    run['model/best'] = neptune.types.File(os.path.join(save_dir, 'models', saveName + '.pth.tar'))
    run.stop()


if __name__ == '__main__':
    main()

