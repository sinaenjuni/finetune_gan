import pytorch_lightning as pl
from argparse import ArgumentParser

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from data_module.cifar10_data_modules import ImbalancedMNISTDataModule

from models.resnet import Resnet_classifier
from models.acgan import ACGAN


def cli_main():
    pl.seed_everything(1234)  # 다른 환경에서도 동일한 성능을 보장하기 위한 random seed 초기화

    parser = ArgumentParser()
    parser.add_argument("--augmentation", default=False, type=bool)
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--imb_factor", default=0.01, type=float)
    parser.add_argument("--balanced", default=False, type=bool)
    parser.add_argument("--retain_epoch_size", default=False, type=bool)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--target_model', type=int, default=99)


    parser = Resnet_classifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args('')
    dm = ImbalancedMNISTDataModule.from_argparse_args(args)

    # model = Resnet_classifier(args.model,
    #                           args.num_class,
    #                           args.sp,
    #                           args.learning_rate,
    #                             args.momentum,
    #                             args.weight_decay,
    #                             args.nesterov,
    #                             args.warmup_epoch,
    #                             args.step1,
    #                             args.step2,
    #                             args.gamma)

    # acgan = ACGAN.load_from_checkpoint(f"/home/sin/git/pytorch.GAN/src/lightning/models/tb_logs/acgan_cifar10_0.01/version_1/checkpoints/epoch={args.target_model}.ckpt")
    # acgan = ACGAN.load_from_checkpoint(f"/home/sin/git/pytorch.GAN/src/lightning/models/tb_logs/acgan_cifar10_0.01_balancing/version_0/checkpoints/epoch={args.target_model}.ckpt")
    # acgan = ACGAN.load_from_checkpoint(f"/home/dblab/git/finetuning_gan/src/train/gan/tb_logs/acgan_cifar10_0.1_False_False/version_0/checkpoints/epoch={args.target_model}.ckpt")
    # acgan = ACGAN.load_from_checkpoint(f"/home/dblab/git/finetuning_gan/src/train/gan/tb_logs/acgan_cifar10_0.01_False_False/version_0/checkpoints/epoch={args.target_model}.ckpt")

    ORI_PATH_100 = {
        "T": "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_original_cifar10_0.01_True_False/version_1/checkpoints/epoch=193-step=2522.ckpt",
        "F": "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_original_cifar10_0.01_False_False/version_1/checkpoints/epoch=183-step=2392.ckpt"
    }

    FINE_PATH_100 = {
        "TT": "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_True_False-gen_True_False/version_1/checkpoints/epoch=99-step=1300.ckpt",
        "TF": "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_True_False-gen_False_False/version_1/checkpoints/epoch=43-step=572.ckpt",
        "FT": "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_False_False-gen_True_False/version_1/checkpoints/epoch=29-step=390.ckpt",
        "FF": "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_False_False-gen_False_False/version_1/checkpoints/epoch=98-step=1287.ckpt"
    }

    # FINE_PATH_10 = {
    #
    # }


    model = Resnet_classifier.load_from_checkpoint(FINE_PATH_100["FF"])


    # checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch:d}_{loss/val:.4}_{acc/val:.4}",
    #     verbose=True,
    #     # save_last=True,
    #     save_top_k=1,
    #     monitor='acc/val',
    #     mode='max',
    # )
    # logger = TensorBoardLogger(save_dir="tb_logs",
    #                            name=f"resnet18_fine_cifar10_{args.imb_factor}_{args.augmentation}_{args.balanced}-gen_True_False",
    #                            default_hp_metric=False
    #                            )
    # logger.log_hyperparams

    trainer = pl.Trainer(accelerator='gpu',
                         gpus=1)

    result = trainer.test(model, dataloaders=dm)
    for name, val in result[0].items():
        print(f'{name}\t{val:.4}')



if __name__ == '__main__':
    cli_main()
