'''
Author: jianzhnie
Date: 2021-12-14 12:03:09
LastEditTime: 2021-12-14 14:09:51
LastEditors: jianzhnie
Description:

'''
import sys
from argparse import ArgumentParser

from pl_bolts.transforms.dataset_normalizations import (cifar10_normalization,
                                                        imagenet_normalization,
                                                        stl10_normalization)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from self_supervised.models.simsiam.simsiam_module import SimSiam

sys.path.append('../')


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # model params
    parser.add_argument('--arch',
                        default='resnet50',
                        type=str,
                        help='convnet architecture')
    # specify flags to store false
    parser.add_argument('--first_conv', action='store_false')
    parser.add_argument('--maxpool1', action='store_false')
    parser.add_argument('--hidden_mlp',
                        default=2048,
                        type=int,
                        help='hidden layer dimension in projection head')
    parser.add_argument('--feat_dim',
                        default=128,
                        type=int,
                        help='feature dimension')
    parser.add_argument('--online_ft', action='store_true')
    parser.add_argument('--fp32', action='store_true')

    # transform params
    parser.add_argument('--gaussian_blur',
                        action='store_true',
                        help='add gaussian blur')
    parser.add_argument('--jitter_strength',
                        type=float,
                        default=1.0,
                        help='jitter strength')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        help='stl10, cifar10')
    parser.add_argument('--data_dir',
                        type=str,
                        default='.',
                        help='path to download data')

    # training params
    # parser.add_argument("--gpus", default=1, type=int, help="num of GPUs")
    parser.add_argument('--num_workers',
                        default=8,
                        type=int,
                        help='num of workers per GPU')
    parser.add_argument('--optimizer',
                        default='adam',
                        type=str,
                        help='choose between adam/lars')
    parser.add_argument('--exclude_bn_bias',
                        action='store_true',
                        help='exclude bn/bias from weight decay')
    parser.add_argument('--warmup_epochs',
                        default=10,
                        type=int,
                        help='number of warmup epochs')
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='batch size per gpu')

    parser.add_argument('--temperature',
                        default=0.1,
                        type=float,
                        help='temperature parameter in training loss')
    parser.add_argument('--weight_decay',
                        default=1e-6,
                        type=float,
                        help='weight decay')
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='base learning rate')
    parser.add_argument('--start_lr',
                        default=0,
                        type=float,
                        help='initial warmup learning rate')
    parser.add_argument('--final_lr',
                        type=float,
                        default=1e-6,
                        help='final learning rate')

    return parser


def main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    seed_everything(1234)

    parser = ArgumentParser()

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    dm = None

    # init datamodule
    if args.dataset == 'stl10':
        dm = STL10DataModule(data_dir=args.data_dir,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0
        args.max_epochs = 3

    # init default datamodule
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule.from_argparse_args(args)
        args.num_classes = dm.num_classes
        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
        args.gpus = 1
        args.max_epochs = 3
    elif args.dataset == 'imagenet':
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = 'lars'
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError(
            'other datasets have not been implemented till now')

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimSiam(**args.__dict__)

    # finetune in real-time
    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(save_last=True,
                                       save_top_k=1,
                                       monitor='val_loss')
    callbacks = [model_checkpoint, online_evaluator
                 ] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer.from_argparse_args(
        args,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
