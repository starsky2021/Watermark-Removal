import os
from random import random

import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class WatermarkDataset(Dataset):
	"""
	WatermarkDataset

	Provides a dataset to train the watermark removal network on. All logos are cross-combined with all images, resulting
	in a dataset of length ||images|| * ||logos||.
	"""

	def __init__(self, images_dir: str, logos_dir: str, image_size=(64, 64)):
		self._image_size=image_size
		self._images_dir = images_dir
		self._images = [p for p in os.listdir(images_dir) if p.endswith(".jpg")]

		assert len(self._images) > 0, "At least one image is required to initialize the dataset"

		self._logos_dir = logos_dir
		self._logos = [p for p in os.listdir(logos_dir) if p.endswith(".png")]

		assert len(self._logos) > 0, "At least one logo is required to initialize the dataset"

		self._randomCrop = torchvision.transforms.RandomCrop(image_size)

	@property
	def image_size(self):
		return self._image_size

	def __len__(self):
		return len(
			self._images)  # Only display each image once every epoch to prevent overfitting // return len(self._images) * len(self._logos)

	def __getitem__(self, item) -> (torch.Tensor, torch.Tensor, torch.Tensor):
		"""
		Fetches the i-th item of the 2-dimensional space where images are on the column and logos are on the row
		:param item:
		:return: The watermarked Image, the original Image and the Watermark
		"""

		image = Image.open(os.path.join(self._images_dir, self._images[item])).convert("RGB")

		if image.size[0] < self._image_size[0] or image.size[1] < self._image_size[1]:
			image = ImageOps.pad(image, self._image_size, Image.BILINEAR)

		image = self._randomCrop(image)

		logo = Image.open(os.path.join(self._logos_dir, self._logos[item % len(self._logos)])).convert("RGBA")
		logo_ratio = logo.size[1] / logo.size[0]
		logo_width = image.size[0] // (1.25 + random() / 2)
		logo = logo.resize([int(x) for x in (logo_width, logo_width * logo_ratio)], Image.BILINEAR)

		random_location = [int(random() * x) for x in (image.size[0] - logo.size[0], image.size[1] - logo.size[1])]

		mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
		mask.paste(logo, random_location, logo)
		mask.putalpha(int(25 + random() * 25))

		image_wm = image.copy()
		image_wm.paste(mask, (0, 0), mask)

		# Reduce the amount of channels to the minimum
		image.convert("RGB")
		image_wm.convert("RGB")

		mask_tensor = TF.to_tensor(mask)[3].gt(0)
		mask_tensor = mask_tensor.to(dtype=torch.int64)

		return TF.to_tensor(image_wm), TF.to_tensor(image), mask_tensor


class WMIDDataset(WatermarkDataset):
	"""
	WatermarkDataset for training the segmentation network

	Overrides the __getitem__ method to only return the watermarked image and the segmentation mask
	"""

	def __getitem__(self, item):
		image_wm, image, mask = super().__getitem__(item)

		return image_wm, mask


class WMGANDataset(WatermarkDataset):
	"""
	WatermarkDataset for training the GAN.

	Yields pairs of dirty and clean images
	"""

	def __getitem__(self, item):
		image_wm, image, mask = super().__getitem__(item)

		return image_wm, image
