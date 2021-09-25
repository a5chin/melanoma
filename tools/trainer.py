from typing import Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path

from lib.models import resnet18, EfficientNet
from lib.dataset import MelaDataset


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = resnet18(pretrained=True)
        self.best_acc = 0
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.writer_train = SummaryWriter(log_dir=self.cfg.MOSWL.LOG_DIR / Path(self.cfg.MODEL.NAME) / Path('train'))
        self.writer_val = SummaryWriter(log_dir=self.cfg.MODEL.LOG_DIR / Path(self.cfg.MODEL.NAME) / Path('val'))

    def train(self):
        traindataset = MelaDataset(train=True)
        traindataloader = DataLoader(
            dataset=traindataset,
            batch_size=self.cfg.DATASET.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        for epoch in range(self.cfg.DATASET.TOTAL_EPOCH):
            self.model.train()
            total, running_loss, running_acc = 0, 0.0, 0.0

            with tqdm(enumerate(traindataloader, 0), total=len(traindataloader)) as pbar:
                pbar.set_description('[Epoch %d/%d]' % (epoch + 1, self.cfg.DATASET.TOTAL_EPOCH))

                for _, data in pbar:
                    images, sexes, ages, sites, labels = data
                    sexes, ages, sites = sexes.view(-1, 1), ages.view(-1, 1), sites
                    images, meta, labels = images.to(self.device), \
                                           torch.cat([sexes, ages, sites], dim=1).to(self.device), \
                                           labels.to(self.device)

                    self.optimizer.zero_grad()

                    preds = self.model(images, meta)
                    loss = self.criterion(preds, labels)
                    loss.backward()

                    self.optimizer.step()

                    results = preds.cpu().detach().numpy().argmax(axis=1)
                    running_acc += accuracy_score(labels.cpu().numpy(), results) * len(images)
                    running_loss += loss.item() * len(images)

                    total += len(images)

                    pbar.set_postfix(OrderedDict(Loss=running_loss / total, Acc=running_acc / total))

                running_acc /= total
                running_loss /= total

                self.writer_train.add_scalar('loss', running_loss, epoch)
                self.writer_train.add_scalar('accuracy', running_acc, epoch)

            self.evaluate(self.model, epoch)

    def evaluate(self, model, epoch: Optional[int] = None):
        valdataset = MelaDataset(train=False)
        valdataloader = DataLoader(
            dataset=valdataset,
            batch_size=self.cfg.DATASET.BATCH_SIZE,
            shuffle=False,
            drop_last=True
        )

        model.eval()
        total, val_loss, val_acc = 0, 0.0, 0.0
        with torch.no_grad():
            for images, labels in valdataloader:
                images = images.to(self.device), labels.to(self.device)
                preds = model(images)
                loss = self.criterion(preds, labels)
                results = preds.cpu().detach().numpy().argmax(axis=1)
                val_acc += accuracy_score(labels.cpu().numpy(), results) * len(labels)
                val_loss += loss.item() * len(labels)

                total += len(labels)

            val_acc /= total
            val_loss /= total

        print('Loss: %f, Accuracy: %f' % (val_loss, val_acc))

        if epoch is not None:
            self.writer_val.add_scalar('loss', val_loss, epoch)
            self.writer_val.add_scalar('accuracy', val_acc, epoch)
            if self.best_acc < val_acc:
                self.best_acc = val_acc
                torch.save(model.state_dict(), self.cfg.MODEL.SAVED_DIR / Path(self.cfg.MODEL.NAME) + '.pth')
