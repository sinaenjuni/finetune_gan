
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import confusion_matrix
from torch.optim import SGD, Adam
from models.base_models.resnet import resnet18, resnet34

import pytorch_lightning as pl

def accNaccPerCls(pred, label, num_class):
    cm = torch.nan_to_num(confusion_matrix(preds=pred, target=label, num_classes=num_class))
    acc = torch.nan_to_num(cm.trace() / cm.sum())
    acc_per_cls = torch.nan_to_num(cm.diagonal() / cm.sum(1))

    return cm, acc, acc_per_cls

class Resnet_classifier(pl.LightningModule):
    def __init__(self,
                 model,
                 num_class,
                 sp,
                 learning_rate,
                 momentum,
                 weight_decay,
                 nesterov,
                 warmup_epoch,
                 step1,
                 step2,
                 gamma,
                 **kwargs):
        super(Resnet_classifier, self).__init__()
        self.save_hyperparameters()


        if model == 'resnet18':
            self.model = resnet18(num_classes=num_class, sp=sp)
        elif model == 'resnet34':
            self.model = resnet34(num_classes=num_class, sp=sp)

        # self.model.fc = nn.Linear(in_features=512, out_features=10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)
        pred = logit.argmax(-1)

        return {"loss":loss, "pred": pred, "label":label}

    def training_epoch_end(self, output):
        loss = torch.stack([x['loss'] for x in output]).mean()
        pred = torch.cat([x['pred'] for x in output])
        label = torch.cat([x['label'] for x in output])

        cm, acc, acc_per_cls = accNaccPerCls(pred=pred, label=label, num_class=self.hparams.num_class)

        self.log("loss/train", loss, logger=True)
        metrics = {"acc/train": acc}
        metrics.update({ f"cls/train/{idx}" : acc for idx, acc in enumerate(acc_per_cls)})
        # cls_dict = { f"{idx}/train" : acc for idx, acc in enumerate(acc_per_cls)}
        # self.logger.experiment.add_scalars("cls", cls_dict)
        # for idx, acc in enumerate(acc_per_cls):
        #     self.logger.experiment.add_scalars(f"cls/{idx}", {"train":acc})
        self.log_dict(metrics, logger=True)

    def validation_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)
        pred = logit.argmax(-1)

        return {"loss":loss, "pred": pred, "label":label}

    def validation_epoch_end(self, output):
        loss = torch.stack([x['loss'] for x in output]).mean()
        pred = torch.cat([x['pred'] for x in output])
        label = torch.cat([x['label'] for x in output])

        cm, acc, acc_per_cls = accNaccPerCls(pred=pred, label=label, num_class=self.hparams.num_class)

        self.log("loss/val", loss, on_epoch=True, logger=True)
        metrics = {"acc/val": acc}
        metrics.update({ f"cls/val/{idx}" : acc for idx, acc in enumerate(acc_per_cls)})
        # cls_dict = { f"{idx}/train" : acc for idx, acc in enumerate(acc_per_cls)}
        # for idx, acc in enumerate(acc_per_cls):
        #     self.logger.experiment.add_scalars(f"cls/{idx}", {"val":acc})

        self.log_dict(metrics, logger=True)



    def test_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)
        pred = logit.argmax(-1)
        return {"loss": loss, "pred": pred, "label": label}

    def test_epoch_end(self, output):
        loss = torch.stack([x['loss'] for x in output]).mean()
        pred = torch.cat([x['pred'] for x in output])
        label = torch.cat([x['label'] for x in output])

        cm, acc, acc_per_cls = accNaccPerCls(pred, label, self.hparams.num_class)
        self.logger.log_hyperparams(params=self.hparams, metrics={"metric(test_acc)": acc})

        metrics = {"loss/test":loss,
                   "acc/test": acc}
        metrics.update({ f"cls/test/{idx}" : acc for idx, acc in enumerate(acc_per_cls)})

        # for idx, acc in enumerate(acc_per_cls):
        #     self.logger.experiment.add_scalars(f"cls/{idx}", {"val": acc})

        self.log_dict(metrics, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = SGD(self.parameters(),
                        momentum=self.hparams.momentum,
                        lr=self.hparams.learning_rate,
                        weight_decay=self.hparams.weight_decay,
                        nesterov=self.hparams.nesterov)

        # def lr_lambda(epoch):
        #     if epoch >= self.hparams.step2:
        #         lr = self.hparams.gamma * self.hparams.gamma
        #     elif epoch >= self.hparams.step1:
        #         lr = self.hparams.gamma
        #     else:
        #         lr = 1
        #     """Warmup"""
        #     if epoch < self.hparams.warmup_epoch:
        #         lr = lr * float(1 + epoch) / self.hparams.warmup_epoch
        #     print("learning_rate", lr)
        #     return lr
        # lr_scheduler = LambdaLR(optimizer, lr_lambda)
        # return [optimizer], [lr_scheduler]

        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--model", default='resnet18', type=str)
        parser.add_argument("--num_class", default=10, type=int)
        parser.add_argument("--sp", default=False, type=bool)

        parser.add_argument('--momentum', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--nesterov', type=bool, default=True)
        parser.add_argument('--warmup_epoch', type=int, default=5)
        parser.add_argument('--step1', type=int, default=160)
        parser.add_argument('--step2', type=int, default=180)
        parser.add_argument('--gamma', type=float, default=0.1)
        return parent_parser








