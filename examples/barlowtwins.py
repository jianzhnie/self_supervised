'''
Author: jianzhnie
Date: 2021-12-13 16:26:57
LastEditTime: 2021-12-14 10:37:07
LastEditors: jianzhnie
Description:

'''
import sys
from argparse import ArgumentParser

from pl_bolts.datamodules import (CIFAR10DataModule, ImagenetDataModule,
                                  STL10DataModule)
from pl_bolts.models.self_supervised.simclr import (SimCLREvalDataTransform,
                                                    SimCLRTrainDataTransform)
from pytorch_lightning import Trainer, seed_everything

from self_supervised.models.barlowtwins.module import BarlowTwins

sys.path.append('../')


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--online_ft',
                        action='store_true',
                        help='run online finetuner')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'imagenet2012', 'stl10'])

    (args, _) = parser.parse_known_args()

    # Data
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--num_workers', default=8, type=int)

    # optim
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.5e-6)
    parser.add_argument('--warmup_epochs', type=float, default=10)

    # Model
    parser.add_argument('--meta_dir',
                        default='.',
                        type=str,
                        help='path to meta.bin for imagenet')

    return parser


def main():
    seed_everything(1234)

    parser = ArgumentParser()

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    dm = None

    # init default datamodule
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule.from_argparse_args(args)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        args.num_classes = dm.num_classes

    elif args.dataset == 'stl10':
        dm = STL10DataModule.from_argparse_args(args)
        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed

        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes

    elif args.dataset == 'imagenet2012':
        dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes

    model = BarlowTwins(**args.__dict__)

    trainer = Trainer.from_argparse_args(args, max_epochs=3, gpus=1)

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
