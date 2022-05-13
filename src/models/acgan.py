
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import confusion_matrix
from torch.optim import SGD, Adam
from models.modules.resnet import resnet18, resnet34
from models.modules.generator import Generator, linear, snlinear, deconv2d, sndeconv2d

import pytorch_lightning as pl
# from torchsummaryX import summary
# import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def accNaccPerCls(pred, label, num_class):
    cm = torch.nan_to_num(confusion_matrix(pred, label, num_classes=num_class))
    acc = torch.nan_to_num(cm.trace() / cm.sum())
    acc_per_cls = torch.nan_to_num(cm.diagonal() / cm.sum(0))

    return cm, acc, acc_per_cls


def d_loss_function(real_logit, fake_logit):
    # real_loss = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
    # fake_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
    real_loss = F.relu(1. - real_logit).mean()
    fake_loss = F.relu(1. + fake_logit).mean()
    d_loss = real_loss + fake_loss
    return d_loss

def g_loss_function(fake_logit):
    # g_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
    g_loss = -fake_logit.mean()
    return g_loss


def cls_loss_function(logit, label):
    cls_loss = F.cross_entropy(logit, label)
    return cls_loss


class ACGAN(pl.LightningModule):
    def __init__(self,
                 model,
                 num_classes,
                 bn,
                 sn,
                 learning_rateD,
                 learning_rateG,
                 image_size,
                 image_channel,
                 std_channel,
                 latent_dim,
                 **kwargs):

        super(ACGAN, self).__init__()
        self.save_hyperparameters()
        self.fixed_noise = torch.randn(10, latent_dim).cuda().repeat(10, 1)

        if sn:
            self.G = Generator(linear=snlinear,
                      deconv=sndeconv2d,
                      image_size=image_size,
                      image_channel=image_channel,
                      std_channel=std_channel,
                               num_classes=num_classes,
                      latent_dim=latent_dim,
                      bn=bn)
        else:
            self.G = Generator(linear=linear,
                      deconv=deconv2d,
                      image_size=image_size,
                      image_channel=image_channel,
                      std_channel=std_channel,
                      latent_dim=latent_dim,
                      bn=bn)

        if model == 'resnet18':
            self.D = resnet18(num_classes=num_classes, sn=sn, discriminator=True)
            # self.D = nn.Sequential(*list(resnet18(num_classes=num_classes, sn=sn).children())[:-2])

        elif model == 'resnet34':
            self.D = resnet34(num_classes=num_classes, sn=sn, discriminator=True)

        # if sn:
            # self.D.add_module("last", FcNAdvModuel(linear=snlinear, feature=512, num_classes=10))
            # self.D.fc = FcNAdvModuel(linear=snlinear, num_classes=num_classes)
        # else:
        #     self.D.add_module("last", FcNAdvModuel(linear=linear, feature=512, num_classes=10))
            # self.D.fc = FcNAdvModuel(linear=linear, num_classes=num_classes)

        # self.cls = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.G(x, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_image, real_label = batch

        # train discriminator
        if optimizer_idx == 0:
            noise = torch.randn(real_image.size(0), 128).cuda()
            fake_label = (torch.rand(real_image.size(0)) * 10).long().cuda()

            fake_image = self(noise, fake_label)
            # self.logger.experiment.add_images(tag="images", img_tensor=fake_image.detach().cpu(),
            #                                   global_step=self.current_epoch)

            real_adv_logit, real_cls_logit = self.D(real_image)
            fake_adv_logit, fake_cls_logit = self.D(fake_image.detach())

            d_adv_loss = d_loss_function(real_adv_logit, fake_adv_logit)
            d_real_cls_loss = cls_loss_function(real_cls_logit, real_label)
            d_fake_cls_loss = cls_loss_function(fake_cls_logit, fake_label)
            # d_loss = (self.la * d_adv_loss) + ((1 - self.la) * d_cls_loss)
            d_loss = d_adv_loss + d_real_cls_loss + d_fake_cls_loss

            return {"loss" : d_loss}
            # return {"loss" : d_adv_loss}

        # train generator
        if optimizer_idx == 1:
            noise = torch.randn(real_image.size(0), 128).cuda()
            fake_label = (torch.rand(real_image.size(0)) * 10).long().cuda()

            fake_image = self(noise, fake_label)
            # fake_logit = self.D(fake_image)

            fake_adv_logit, fake_cls_logit = self.D(fake_image)
            g_adv_loss = g_loss_function(fake_adv_logit)
            g_cls_loss = cls_loss_function(fake_cls_logit, fake_label)
            g_loss = g_adv_loss + g_cls_loss
            # g_loss = (self.la * g_adv_loss) + ((1 - self.la) * g_cls_loss)

            return {"loss": g_loss}
            # return {"loss": g_adv_loss}


    def training_epoch_end(self, output):
        # for i, v in enumerate(output):
        #     print(i, v)

        d_loss = torch.stack([x[0]['loss'] for x in output]).mean()
        g_loss = torch.stack([x[1]['loss'] for x in output]).mean()
        self.log_dict({"loss/d": d_loss, "loss/g": g_loss}, logger=True)


    def validation_step(self, batch, batch_idx):
        image, label = batch
        adc_logit, cls_logit = self.D(image)
        loss = cls_loss_function(cls_logit, label)
        pred = cls_logit.argmax(-1)

        return {"loss": loss, "pred": pred, "label": label}

    def validation_epoch_end(self, output):
        loss = torch.stack([x['loss'] for x in output]).mean()
        pred = torch.cat([x['pred'] for x in output])
        label = torch.cat([x['label'] for x in output])

        cm, acc, acc_per_cls = accNaccPerCls(pred=pred, label=label, num_class=self.hparams.num_classes)

        self.log("loss/val", loss, on_epoch=True, logger=True)
        metrics = {"acc/val": acc}
        metrics.update({ f"cls/val/{idx}" : acc for idx, acc in enumerate(acc_per_cls)})
        # cls_dict = { f"{idx}/train" : acc for idx, acc in enumerate(acc_per_cls)}
        # for idx, acc in enumerate(acc_per_cls):
        #     self.logger.experiment.add_scalars(f"cls/{idx}", {"val":acc})

        self.log_dict(metrics, logger=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        # logit = self(image)
        adc_logit, cls_logit = self.D(image)

        loss = cls_loss_function(cls_logit, label)
        pred = cls_logit.argmax(-1)
        return {"loss": loss, "pred": pred, "label": label}

    def test_epoch_end(self, output):
        loss = torch.stack([x['loss'] for x in output]).mean()
        pred = torch.cat([x['pred'] for x in output])
        label = torch.cat([x['label'] for x in output])

        cm, acc, acc_per_cls = accNaccPerCls(pred, label, self.hparams.num_classes)
        self.logger.log_hyperparams(params=self.hparams, metrics={"metric(test_acc)": acc})

        metrics = {"loss/test":loss,
                   "acc/test": acc}
        metrics.update({ f"cls/test/{idx}" : acc for idx, acc in enumerate(acc_per_cls)})

        # for idx, acc in enumerate(acc_per_cls):
        #     self.logger.experiment.add_scalars(f"cls/{idx}", {"val": acc})

        self.log_dict(metrics, logger=True)


    def configure_optimizers(self):
        d_optimizer = Adam(self.D.parameters(),
                           lr=self.hparams.learning_rateD,
                           weight_decay=self.hparams.weight_decay,
                           betas=(self.hparams.beta1, self.hparams.beta2))
        g_optimizer = Adam(self.G.parameters(),
                           lr=self.hparams.learning_rateG,
                           weight_decay=self.hparams.weight_decay,
                           betas=(self.hparams.beta1, self.hparams.beta2))

        return [d_optimizer, g_optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--model", default='resnet18', type=str)
        parser.add_argument("--image_size", default=32, type=int)
        parser.add_argument("--image_channel", default=3, type=int)
        parser.add_argument("--std_channel", default=64, type=int)
        parser.add_argument("--latent_dim", default=128, type=int)
        parser.add_argument('--learning_rateD', type=float, default=4e-4)
        parser.add_argument('--learning_rateG', type=float, default=1e-4)
        parser.add_argument("--num_classes", default=10, type=int)
        parser.add_argument("--sn", default=True, type=bool)
        parser.add_argument("--bn", default=True, type=bool)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--beta2", default=0.999, type=float)
        parser.add_argument("--la", default=0.3, type=float)

        parser.add_argument('--weight_decay', type=float, default=1e-5)
        return parent_parser





