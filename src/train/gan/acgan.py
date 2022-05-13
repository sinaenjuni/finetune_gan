
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import confusion_matrix
from torch.optim import SGD, Adam
from models.acgan import ACGAN

import pytorch_lightning as pl
# from torchsummaryX import summary
# import matplotlib.pyplot as plt
from torchvision.utils import make_grid




def cli_main():
    from argparse import ArgumentParser
    from data_module.cifar10_data_modules import  ImbalancedMNISTDataModule
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies.ddp import DDPStrategy

    pl.seed_everything(1234)  # 다른 환경에서도 동일한 성능을 보장하기 위한 random seed 초기화

    parser = ArgumentParser()
    # data amount
    parser.add_argument("--augmentation", default=True, type=bool)
    parser.add_argument("--balanced", default=False, type=bool)
    parser.add_argument("--imb_factor", default=0.01, type=float)
    #
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--retain_epoch_size", default=False, type=bool)
    parser.add_argument('--epoch', type=int, default=200)


    parser = ACGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args('')
    dm = ImbalancedMNISTDataModule.from_argparse_args(args)

    model = ACGAN(**vars(args))
    # summary(model, x=torch.rand(10, 128))



    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch:d}",
                                                        verbose=True,
                                                       every_n_epochs=10,
                                                        # save_last=True,
                                                        save_top_k=-1,
                                                        # monitor='acc/val',
                                                        # mode='max',
    )

    logger = TensorBoardLogger(save_dir="tb_logs",
                               name=f"acgan_cifar10_{args.imb_factor}_{args.augmentation}_{args.balanced}",
                               default_hp_metric=False
                               )
    # logger.experiment.add_images()

    trainer = pl.Trainer(max_epochs=args.epoch,
                         # callbacks=[EarlyStopping(monitor='val_loss')],
                         callbacks=[checkpoint_callback],
                         strategy=DDPStrategy(find_unused_parameters=True),
                         accelerator='gpu',
                         gpus=-1,
                         logger=logger
                         )
    trainer.fit(model, datamodule=dm)

    result = trainer.test(model, dataloaders=dm.test_dataloader())

    print(result)


if __name__ == '__main__':
    cli_main()




