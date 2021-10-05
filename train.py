import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import (AdamW, BertConfig, BertForSequenceClassification,
                          BertTokenizerFast,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizerFast)

from data import AmazonTrainDataset
from utils import Logger, get_lr


# 训练器类
class Trainer(object):

    def __init__(self, model, train_loader, val_loader, train_size, val_size,
                 optimizer, scheduler, name, multi_gpu=False):

        # gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.multi_gpu = multi_gpu

        # data
        self.train_loader, self.val_loader = train_loader, val_loader
        self.train_size, self.val_size = train_size, val_size
        self.batch_size = train_loader.batch_size

        # model spec
        self.model = model.to(self.device)
        if self.multi_gpu:
            self.model = nn.DataParallel(model, device_ids=[0, 1])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.name = name

        # performance
        self.best_val_acc = 0.0
        self.best_val_loss = 5.0

        # logger
        if not os.path.exists('record'):
            os.mkdir('record')
        self.train_logger = Logger('record/train.log', ['epoch', 'loss', 'acc', 'lr'])
        self.val_logger = Logger('record/val.log', ['epoch', 'loss', 'acc'])

    def train(self, num_epochs):
        self.num_epochs = num_epochs
        self.step = 0

        # 训练num_epochs轮
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            # 训练一轮
            train_loss, train_acc = self._train()
            # 验证
            val_loss, val_acc = self._val()

            # log
            self.train_logger.log({
                'epoch': self.epoch, 'loss': train_loss, 'acc': train_acc, 'lr': get_lr(self.optimizer)
            })
            self.val_logger.log({
                'epoch': self.epoch, 'loss': val_loss, 'acc': val_acc
            })

            print('Epoch: {}'.format(epoch))
            print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(train_loss, train_acc))
            print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))

            # 如果验证集准确率有所提高，保存模型
            if val_acc > self.best_val_acc or (val_acc == self.best_val_acc and val_loss < self.best_val_loss):
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.save_model()

        # 输出训练最优结果到文件
        print('Best val acc: {:.4f}'.format(self.best_val_acc))
        with open('record/best_val_acc.txt', 'w') as f:
            f.write('Model\tbest acc\tloss\n')
            f.write(f'{self.name}\t{self.best_val_acc}\t{self.best_val_loss}\n')

    def save_model(self):
        r'''save model state dict'''
        if self.multi_gpu:
            torch.save(self.model.module.state_dict(), f'record/{self.name}.pth')
        else:
            torch.save(self.model.state_dict(), f'record/{self.name}.pth')

    def _train(self):
        r''' one epoch train '''
        self.model.train()

        running_loss = 0.0
        running_corrects = 0.0

        iter_count = 0
        # mini-batch 训练
        for input_ids, attention_masks, labels in self.train_loader:
            # 数据加载到GPU
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            # forward
            loss, logits = self.model(
                input_ids,
                attention_mask=attention_masks,
                labels=labels)

            # backward
            self.optimizer.zero_grad()
            if self.multi_gpu:
                loss.sum().backward()
                loss = loss.mean()
            else:
                loss.backward()
            self.optimizer.step()

            # 统计损失、准确率
            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct_count = int((preds == labels).sum())
            running_corrects += correct_count

            print('Step {}, Epoch {}/{}, Iter {}/{}, Loss {:.4f}, Acc {:.4f}.'.format(
                self.step, self.epoch, self.num_epochs - 1,
                iter_count, int(self.train_size / self.batch_size),
                loss.item(), correct_count / labels.size(0)))

            iter_count += 1
            self.step += 1

        # end one epoch
        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss / self.train_size, running_corrects / self.train_size

    @torch.no_grad()
    def _val(self):
        r'''validate on validation set'''
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0.0

        # 此部分与_train类似，只需前向传播
        for input_ids, attention_masks, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            # forward
            loss, logits = self.model(
                input_ids,
                attention_mask=attention_masks,
                labels=labels)

            if self.multi_gpu:
                loss = loss.mean()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct_count = int((preds == labels).sum())
            running_corrects += correct_count

        return running_loss / self.val_size, running_corrects / self.val_size


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(42)

    ## args
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers for dataloader')
    parser.add_argument('-m', '--model', type=str, default='distilbert', choices=['bert', 'distilbert'])
    parser.add_argument('--multi_gpu', action='store_true', help='using multiple GPUs')
    args = parser.parse_args()
    print(args)

    ## model
    if args.model == 'bert':
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, output_attentions=False,
                                                              output_hidden_states=False, return_dict=False)
    elif args.model == 'distilbert':
        model_name = 'distilbert-base-uncased'
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, output_attentions=False,
                                                                    output_hidden_states=False, return_dict=False)

    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model.num_labels = 5
    # model.load_state_dict(torch.load('final_result/distilbert.pth'))
    print(model)

    ## data
    dataset = AmazonTrainDataset(tokenizer, model_name, 'dataset/Video_Games_5.json')
    train_size = int(0.9 * len(dataset));
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    ## train spec
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5])
    trainer = Trainer(model, train_loader, val_loader, train_size, val_size, optimizer, scheduler, args.model,
                      args.multi_gpu)
    trainer.train(9)
