import torch
import torchvision

from torch.nn import functional as F

from pytorch_lightning.core.lightning import LightningModule


class WMIDNet(LightningModule):
	"""
	WaterMark IDentification Network

	Identifies watermarks on an input image. Returns a mask of pixels that show the location of the watermark.

	Wraps a DeepLabV3 model to perform this task.
	"""

	def __init__(self, lr=0.005):
		super().__init__()
		self.lr = lr
		self.wrap = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

	def forward(self, image_wm):
		return self.wrap(image_wm)["out"]

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y)

		return {"loss": loss}

	def configure_optimizers(self):
		return torch.optim.Adam(
			self.parameters(),
			lr=self.lr
		)

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		return {"val_loss": F.cross_entropy(y_hat, y)}

	def validation_epoch_end(self, outputs):
		val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

		return {"val_loss": val_loss_mean}


