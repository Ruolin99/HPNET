from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import ArgoverseV1Dataset
from torch_geometric.loader import DataLoader
from model import HPNet

from transforms import LaneRandomOcclusion

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    HPNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = HPNet.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    trainer = pl.Trainer(devices=args.devices, accelerator='gpu')
    test_dataset = ArgoverseV1Dataset(args.root, 'test', transform=LaneRandomOcclusion(0.0))
    dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,num_workers=args.num_workers, pin_memory=args.pin_memory,persistent_workers=args.persistent_workers)
    trainer.test(model, dataloader)

