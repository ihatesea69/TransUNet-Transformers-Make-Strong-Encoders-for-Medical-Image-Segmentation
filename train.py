import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from experiment_utils import apply_attention_config, build_attention_suffix, parse_attention_scales
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./splits/synapse', help='split dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,
                    default=8, help='dataloader worker count')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--max_train_samples', type=int,
                    default=0, help='limit training samples for smoke runs; 0 uses the full dataset')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--attention_mode', type=str,
                    default='none', choices=['none', 'pre_hidden', 'cnn_fusion'],
                    help='where to inject CNN attention before the transformer')
parser.add_argument('--attention_scales', type=str,
                    default='',
                    help='comma-separated CNN scales, e.g. 1/8,1/4,1/2')
parser.add_argument('--attention_reduction', type=int,
                    default=16, help='channel reduction used by the CNN attention blocks')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    train_data_dir = os.environ.get('TRANSUNET_TRAIN_DATA_DIR', './data/Synapse/train_npz')
    weights_dir = os.environ.get('TRANSUNET_WEIGHTS_DIR', None)
    dataset_config = {
        'Synapse': {
            'root_path': train_data_dir,
            'list_dir': './splits/synapse',
            'num_classes': 9,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.attention_scales = parse_attention_scales(args.attention_mode, args.attention_scales)
    if args.attention_mode != 'none' and 'R50' not in args.vit_name:
        raise ValueError('CNN attention modes require a hybrid R50-ViT backbone.')
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    model_dir = os.environ.get('TRANSUNET_MODEL_DIR', './model')
    snapshot_path = "{}/{}/{}".format(model_dir, args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + build_attention_suffix(
        args.attention_mode,
        args.attention_scales,
        args.attention_reduction,
    )

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    apply_attention_config(
        config_vit,
        mode=args.attention_mode,
        scales=args.attention_scales,
        reduction=args.attention_reduction,
    )
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    if weights_dir:
        weight_file = os.path.join(weights_dir, 'R50+ViT-B_16.npz')
    else:
        weight_file = config_vit.pretrained_path
    net.load_from(weights=np.load(weight_file))

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)
