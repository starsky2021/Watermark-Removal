#!/usr/bin/env python3


import argparse

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from data import WMIDDataset, WMGANDataset
from networks.gan import GAN


def get_args():
	parser = argparse.ArgumentParser(description='Train the model')
	parser.add_argument(
		'-x', "--input-logos",
		type=str,
		required=True,
		help="Input directory containing logos (SVG format)"
	)
	parser.add_argument(
		'-i', "--input-images",
		type=str,
		required=True,
		help="Input directory containing images (PIL-supported encoding)"
	)
	parser.add_argument(
		'-o', "--output",
		type=str,
		required=True,
		help="Output directory for the model .pth files"
	)
	parser.add_argument(
		'-m', '--model',
		type=str,
		required=False,
		help='Model .pth file to use as starting point'
	)
	parser.add_argument(
		'-b', '--batch-size',
		type=int,
		default=2,
	)
	parser.add_argument(
		'-e', '--epochs',
		type=int,
		default=100,
	)
	parser.add_argument(
		'-s', '--split',
		type=float,
		default=0.1,
		help="Coefficient for which fraction of the data should be reserved for validation"
	)

	return parser.parse_args()


def get_data(input_images: str, input_logos: str,
			 split: float = 0.25, batch_size=4, workers: int = 4) -> (DataLoader, DataLoader):
	data = WMGANDataset(input_images, input_logos)
	data_train, data_val = random_split(data, [int(round(len(data) * n)) for n in (1 - split, split)])

	# Define two DataLoaders: one for the training set and one for the validation set
	[loader_train, loader_val] = [
		DataLoader(
			ds,
			batch_size=batch_size,
			shuffle=ds == data_train,
			num_workers=workers,
			pin_memory=False
		)
		for ds in (data_train, data_val)
	]

	return loader_train, loader_val


if __name__ == "__main__":
	args = get_args()

	assert 1.0 > args.split >= 0.0, "Validation split value must be between 0.0 and 1.0"
	assert args.batch_size > 1, "Batch size must be at least 1"

	print(f"Starting with args: {args}")

	loader_train, loader_val = get_data(args.input_images, args.input_logos, args.split, args.batch_size)

	input_nc = 4
	output_nc = 4

	model = GAN()
	trainer = Trainer(
		gpus=1,
		auto_lr_find=False,
		default_root_dir='./checkpoints'
	)
	trainer.fit(model, loader_train, loader_val)
