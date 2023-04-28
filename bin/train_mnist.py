import os, sys
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main(args):

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device_str = "cuda:%d"%(args.gpu_id)
    else:
        device_str = "cpu"

    print("="*50)
    print("Get system info:")
    print("Scheduled affinity (cpu): ",os.sched_getaffinity(0))
    if torch.cuda.is_available():
        print("gpu count: ",torch.cuda.device_count())
        print("Current device: ",torch.cuda.current_device())
        print("Device: ",torch.cuda.get_device_name())
    print("="*50)

    sys.path.append(args.srbm_path)
    import RBM

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(device_str)
    init_cond = {'w_sig':args.w_sig, 'm':args.m, 'sig':args.sig, 'm_scheme':args.m_scheme}
    rbm = RBM.SRBM(n_v=784, n_h=args.n_h, k=args.k, init_cond=init_cond, device=device_str)
    
    train_ds = datasets.MNIST(args.data_path, \
            train=True, \
            download=True, \
            transform=transforms.Compose([transforms.ToTensor()]))

    num_workers = RBM.check_num_workers(train_ds, args.batch_size, verbose=True)
    dl_kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {}
    train_dl = DataLoader(\
            dataset = train_ds, \
            batch_size = args.batch_size, \
            drop_last = True, \
            shuffle = True, \
            **dl_kwargs)

    print("Start training... on %s"%(rbm.device))

    history = rbm.fit(train_dl, args.epoch, args.lr, verbose=False, lr_decay=args.lr_decay)

    print("Training finished!")

    print("Saving model %s"%(args.model_path+rbm.name))
    rbm.save(args.model_path)
    print("Model saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRBM train MNIST")
    parser.add_argument('--srbm_path', type=str, default='../RBM', help='./SRBM path')
    parser.add_argument('--data_path', type=str, default='../data', help='data path')
    parser.add_argument('--model_path', type=str, default='../model', help='model path')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--k', type=int, default=1, help='k')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--w_sig', type=float, default=0.1, help='initial weight width')
    parser.add_argument('--m', type=float, default=12, help='initial RBM mass')
    parser.add_argument('--sig', type=float, default=1., help='hidden layer sigma')
    parser.add_argument('--m_scheme', type=int, default=0, help='mass scheme: 0-fixed, 1-global, 2-local')
    parser.add_argument('--n_h', type=int, default=784, help='number of hidden nodes')
    args = parser.parse_args()

    main(args)
