import argparse
import datetime
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils import data

from model import RELIABLE
import utils
from train import train_one_epoch
from dataset_loader import load_dataset, IncompleteDatasetSampler

warnings.filterwarnings("ignore")

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training')

    # config path
    parser.add_argument('--config_file', type=str, default=None)

    # backbone parameters
    parser.add_argument('--encoder_dim', type=list, nargs='+', default=[])
    parser.add_argument('--embed_dim', type=int, default=0)

    # model parameters
    parser.add_argument('--temperature', type=float, default=0.5)
    # parser.add_argument('--start_rectify_epoch', type=int, default=100)
    parser.add_argument('--start_rectify_epoch', type=int, default=100)  # rectify from begin
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--n_views', type=int, default=2, help='number of views')
    parser.add_argument('--n_samples', type=int, default=None, help='number of samples')  # 样本数量
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')

    # training setting
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=20, help='epochs to warmup learning rate')
    parser.add_argument('--data_norm', type=str, default='standard', choices=['standard', 'min-max', 'l2-norm'])
    parser.add_argument('--train_time', type=int, default=5)

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Initial value of the weight decay. (default: 0)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    # data loader and logger
    parser.add_argument('--dataset', type=str, default='LandUse21',
                        choices=['LandUse21', 'Scene15', ])
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='./',
                        help='path to your folder of dataset')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='path where to save, empty for no saving')

    parser.add_argument('--print_freq', default=10)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # 新加入的用于计算互补性损失函数的超参数
    parser.add_argument("--k", default=5, type=int)
    # 互补性损失函数超参数
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--calculate_W_every_epoch", default=False, type=bool)

    # Using GPU to accelerate the data loading
    parser.add_argument("--accelerate", type=str, default='yes', choices=['yes', 'no'])

    parser.set_defaults(pin_mem=True)

    return parser


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)

    device = torch.device(args.device)

    dataset = load_dataset(args)
    dataset_train, dataset_test = dataset, dataset

    sampler_train = IncompleteDatasetSampler(dataset_train, seed=args.seed)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    # Used for fusion
    if args.missing_rate >= 0.0:
        temp_missing_rate = args.missing_rate
        args.missing_rate = 0.0
        dataset_all = load_dataset(args)
        sampler_train_all = IncompleteDatasetSampler(dataset_all, seed=args.seed)
        args.missing_rate = temp_missing_rate

    # Using GPU to accelerate the data loading
    if args.accelerate == 'yes':
        args.num_workers = 0

    if args.batch_size > len(sampler_train):
        args.batch_size = len(sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    data_loader_train_all = torch.utils.data.DataLoader(
        dataset_all,
        sampler=sampler_train_all,
        batch_size=args.n_samples,
        shuffle=False,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = RELIABLE(n_views=args.n_views,
                   n_samples=args.n_samples,
                   layer_dims=args.encoder_dim,
                   temperature=args.temperature,
                   n_classes=args.n_classes,
                   drop_rate=args.drop_rate, )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.train_id == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write('Batch size: {}'.format(args.batch_size))
        state_logger.write('Start time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        state_logger.write('Train parameters: {}'.format(args).replace(', ', ',\n'))
        state_logger.write(model.__repr__())
        state_logger.write(optimizer.__repr__())
        print('Data loaded: there are {:} samples.'.format(len(dataset_train)))

    state_logger.write('\n>> Start training {}-th initial, seed: {},'.format(args.train_id, args.seed))

    best_result = {'nmi': 0.0, 'ari': 0.0, 'f': 0.0, 'acc': 0.0}
    # best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs or epoch == 0
        train_state = train_one_epoch(
            model,
            data_loader_train,
            data_loader_train_all,
            data_loader_test,
            optimizer,
            device, epoch,
            # scheduler,
            state_logger,
            args,
        )
        if args.print_this_epoch:
            state_logger.write('Epoch {} K-means: NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'
                               .format(epoch, train_state['nmi'], train_state['ari'], train_state['f'],
                                       train_state['acc']))

        if epoch >= 100 and train_state['nmi'] >= best_result['nmi'] and train_state['ari'] >= best_result['ari'] and train_state['f'] >= best_result['f'] and train_state['acc'] >= best_result['acc']:
            best_result['nmi'] = train_state['nmi']
            best_result['ari'] = train_state['ari']
            best_result['f'] = train_state['f']
            best_result['acc'] = train_state['acc']

        if args.output_dir and epoch + 1 == args.epochs:
            state_logger.write('Best Result: K-means: NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'
                               .format( best_result['nmi'], best_result['ari'], best_result['f'],
                                       best_result['acc']))
            # torch.save(model, args.output_dir + f"checkpoint_{epoch}")

    return train_state, best_result

def main(args):
    start_time = time.time()

    result_avr = {'nmi': [], 'ari': [], 'f': [], 'acc': []}

    batch_scale = args.batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_scale

    state_logger = utils.FileLogger(os.path.join(args.output_dir, 'log_train.txt'))  # noting the state

    for t in range(args.train_time):
        args.train_id = t
        train_state, best_result = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        for k, v in best_result.items():
            result_avr[k].append(v)

    for k, v in result_avr.items():
        x = np.asarray(v) * 100
        result_avr[k] = [x.mean(), x.std()]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
    state_logger.write('Average K-means Result: ACC = {:.2f}({:.2f}) NMI = {:.2f}({:.2f}) ARI = {:.2f}({:.2f})'
                       .format(*result_avr['acc'], *result_avr['nmi'], *result_avr['ari']))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

    folder_name = '_'.join(
        [args.dataset, 'msrt', str(args.missing_rate),
         'tau', str(args.temperature), 'bs', str(args.batch_size), 'blr', str(args.blr)])

    args.embed_dim = args.encoder_dim[0][-1]
    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'visualize')).mkdir(parents=True, exist_ok=True)

    main(args)
