import json
import os, time
import os.path as op
from datetime import datetime
import cv2

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import nn
import torchvision
from torchvision.utils import save_image


def _g(w, h, x, y, c, n, i):
    return ((c * h + y) * w + x) * n + i


def export_fm(fm, out_pt):
    print(out_pt)
    f = open(out_pt, "w")

    if len(fm.shape) == 4:
        h = fm.shape[2]
        w = fm.shape[3]
        nImgs = fm.shape[0]
        channels = fm.shape[1]
    elif len(fm.shape) == 3:
        h = 1
        w = fm.shape[2]
        nImgs = fm.shape[0]
        channels = fm.shape[1]
    elif len(fm.shape) == 2:
        h = 1
        w = 1
        nImgs = fm.shape[0]
        channels = fm.shape[1]
    else:
        channels = fm.shape[0]
        h = 1
        w = 1
        nImgs = 1

    print("size", w, h, channels, nImgs, file=f)
    for c in range(channels):
        for y in range(h):
            for x in range(w):
                for n in range(nImgs):
                    if len(fm.shape) == 4:
                        it = fm[n, c, y, x].item()
                    elif len(fm.shape) == 3:
                        it = fm[n, c, x].item()
                    elif len(fm.shape) == 2:
                        it = fm[n, c].item()
                    else:
                        it = fm[c].item()
                    print("{:.6e}".format(it), file=f, end=" ")


def Dataloader_by_Index(data_loader, target=0):
    print(target)
    for index, data in enumerate(data_loader, target):
        print(index)
        return data
    return None


def go_train(train_loader, config, recognizer, optimizer, loss, save_im_pt, e):
    train_loss = 0.0
    pbar = tqdm(train_loader)
    for idx, mb in enumerate(pbar):
        # mb = Dataloader_by_Index(train_loader, torch.randint(len(train_loader), size=(1,)).item())
        inp = mb['image'].cuda()
        lbl = mb['label'].cuda()

        out = recognizer(inp)

        _, preds = torch.max(out.data, 1)

        if idx == 0 and config["save_images"]:
            save_image(inp, os.path.join(save_im_pt, 'out_train' + str(e) + '.png'))

        cur_loss = loss(out, lbl)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        train_loss += cur_loss.item()

        # print("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))
        pbar.set_description("epoch %d Train [Loss %.6f]" % (e, cur_loss.item()))

    return train_loss


def go_test(test_loader, config, recognizer, loss, save_im_pt, e):
    val_corrects, tp, fp, fn, tn = 0, 0, 0, 0, 0
    total_fake_lbls, total_real_lbls = 0, 0
    test_loss = 0.0
    pbar = tqdm(test_loader)
    for idx, mb in enumerate(pbar):
        inp = mb['image'].cuda()
        lbl = mb['label'].cuda()

        out = recognizer(inp)

        _, preds = torch.max(out.data, 1)

        if idx == 0 and config["save_images"]:
            save_image(inp, os.path.join(save_im_pt, 'out_test' + str(e) + '.png'))

        cur_loss = loss(out, lbl)

        test_loss += cur_loss.item()

        fake_lbls = lbl.data == 0
        real_lbls = lbl.data == 1

        total_real_lbls += torch.sum(real_lbls)
        total_fake_lbls += torch.sum(fake_lbls)

        tn += torch.sum(preds[fake_lbls] == lbl.data[fake_lbls]).to(torch.float32)
        tp += torch.sum(preds[real_lbls] == lbl.data[real_lbls]).to(torch.float32)
        fn += torch.sum(preds[fake_lbls] != lbl.data[fake_lbls]).to(torch.float32)
        fp += torch.sum(preds[real_lbls] != lbl.data[real_lbls]).to(torch.float32)

        val_corrects += torch.sum(preds == lbl.data).to(torch.float32)
        pbar.set_description("epoch %d Test [Loss %.6f]" % (e, cur_loss.item()))
    print('-----------------------')
    print('Total_fake: ', total_fake_lbls, 'Total_real: ', total_real_lbls)
    print('Correct: ', val_corrects)
    print('FP:', fp.item(), 'TP:', tp.item(), 'FN:', fn.item(), 'TN:', tn.item())
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print('Recall: ', recall, 'Precision: ', precision)
    print('-----------------------')
    return test_loss, recall, precision


def run_training(config, recognizer, optimizer, train_dataset, test_dataset, valid_dataset, my_dataset, save_pt,
                 save_im_pt,
                 start_ep):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['minibatch_size'],
                              shuffle=True,
                              num_workers=10)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1024,
                             shuffle=True,
                             num_workers=10)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1024,
                              shuffle=True,
                              num_workers=10)

    my_loader = DataLoader(dataset=my_dataset,
                            batch_size=1024,
                            shuffle=True,
                            num_workers=10)

    loss = nn.CrossEntropyLoss()

    min_test_loss = np.inf
    stat = {'epochs': [],
            'train_losses': [],
            'valid_losses': [],
            'test_losses': [],
            'recall': [],
            'precision': []}

    for e in range(start_ep, config['epoch_num']):
        # pbar = tqdm(train_loader)
        # for idx, mb in enumerate(pbar):

        start_time = time.time()

        train_loss = go_train(train_loader, config, recognizer, optimizer, loss, save_im_pt, e)
        #
        # valid_loss, midv_acc = go_test(valid_loader, config, recognizer, loss, save_im_pt, e)
        # print('MIDV: ', midv_acc)

        test_loss, recall, precision = go_test(test_loader, config, recognizer, loss, save_im_pt, e)

        # _, my_res = go_test(my_loader, config, recognizer, loss, save_im_pt, e)
        # print('FaceForensics:', my_res)

        print(
            f'Epoch {e} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {test_loss / len(test_loader)}')

        if min_test_loss > test_loss:
            print(
                f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss / len(test_loader):.6f}) \t Saving The Model')

        ep_save_pt = op.join(save_pt, str(e))
        if not os.path.exists(ep_save_pt):
            os.mkdir(ep_save_pt)

        print('Epoch {} -> Training Loss({:.2f} sec): {}'.format(e, time.time() - start_time,
                                                                 train_loss / len(train_loader)))

        torch.save({
            'epoch': e,
            'model_state_dict': recognizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
        }, op.join(ep_save_pt, "model.pt"))

        stat['epochs'].append(e)
        stat['train_losses'].append(train_loss / len(train_loader))
        # stat['valid_losses'].append(valid_loss / len(valid_loader))

        stat['test_losses'].append(test_loss / len(test_loader))
        stat['recall'].append(recall.item())
        stat['precision'].append(precision.item())

        plt.figure(figsize=(12, 7))
        plt.xlabel("Epoch", fontsize=18)

        plt.plot(stat['epochs'], stat['train_losses'], 'o-', label='train loss', ms=4)
                 # , alpha=0.7, label='0.01', lw=5, mec='b', mew=1, ms=7)
        plt.plot(stat['epochs'], stat['test_losses'], 'o-.', label='test loss', ms=4)
                   # , alpha=0.7, label='0.1', lw=5, mec='b', mew=1, ms=7)
        # plt.plot(stat['epochs'], stat['valid_losses'], 'o-.', label='valid loss',
        #          ms=4)  # , alpha=0.7, label='0.1', lw=5, mec='b', mew=1, ms=7)
        plt.plot(stat['epochs'], stat['recall'], 'o--', label='Max recall:' + str(max(stat['recall'])), ms=4)
                 # , alpha=0.7, label='0.3', lw=5, mec='b', mew=1, ms=7)
        plt.plot(stat['epochs'], stat['precision'], 'o--', label='Max precision:' + str(max(stat['precision'])), ms=4)

        plt.legend(fontsize=18,
                   ncol=2,  # количество столбцов
                   facecolor='oldlace',  # цвет области
                   edgecolor='black',  # цвет крайней линии
                   title='value',  # заголовок
                   title_fontsize='18'  # размер шрифта заголовка
                   )
        plt.grid(True)
        plt.savefig(op.join(ep_save_pt, 'graph.png'))

        with open(op.join(ep_save_pt, 'info.txt'), 'w') as info_txt:
            info_txt.write(config['description'] + '\n')
            for el in zip(stat['epochs'], stat['recall'], stat['precision']):
                info_txt.write(str(el[0]) + ' ' + str(el[1]) + ' ' + str(el[2]) + '\n')


def prepare_dirs(config):
    save_paths = {}
    files_to_start = {}
    from_file = False
    start_ep = 0
    checkpoint_pt = config["checkpoint_pt"]
    images_pt = config["images_pt"]
    if not op.exists(checkpoint_pt):
        os.mkdir(checkpoint_pt)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")

    # if config['file_to_start'] != "":
    #     dir = config['file_to_start'].split("/")[0]
    #     chpnt = config['file_to_start'].split("/")[1]
    #     start_ep = int(chpnt.split(".")[0]) + 1
    #
    #     # assert sum([op.exists(pt) for pt in files_to_start.values()]) == 4
    #     from_file = True
    # else:
    save_pt = op.join(checkpoint_pt, dt_string)
    save_im_pt = op.join(images_pt, dt_string)
    if not op.exists(save_pt):
        os.makedirs(save_pt)
    if not op.exists(save_im_pt):
        os.makedirs(save_im_pt)

    return save_pt, save_im_pt
