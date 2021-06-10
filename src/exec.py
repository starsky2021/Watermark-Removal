#!/usr/bin/env python3

import argparse
import logging


def get_args():
	parser = argparse.ArgumentParser(description='Execute the trained model on an input image')
	parser.add_argument(
		'-i', "--input",
		type=str,
		required=True,
		help="Input .png file"
	)
	parser.add_argument(
		'-o', "--output",
		type=str,
		required=True,
		help="Output .png file"
	)
	parser.add_argument(
		'-m', '--model',
		type=str,
		required=True,
		help='Model .pth file'
	)

	return parser.parse_args()


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	args = get_args()

	logging.info(f"Removing the watermark from {args.input} using the model {args.model}")

	# Todo: remove the watermark using the trained model

	logging.info(f"Saving the cleaned image as {args.output}")
