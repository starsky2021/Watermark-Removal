import torch
import torchvision

from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn.functional as F
import math
from skimage.measure import compare_ssim



# leaky ReLU slope 0.2

class GAN(LightningModule):
	"""The complete GAN compiled in a single LightningModule"""

	def __init__(self, input_channels=3, output_channels=3, target_real_label=1., target_fake_label=0.):
		super().__init__()

		self.model_g = Generator(input_channels, output_channels, use_dropout=True)
		self.model_d = Discriminator(input_channels + output_channels, use_sigmoid=True)

		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))

	def gan_loss(self, y_hat, is_real):
		y = (self.real_label if is_real else self.fake_label).expand_as(y_hat)

		return F.binary_cross_entropy(y_hat, y)  # Lsgan: use MSEloss?

	def forward(self, z):
		return self.model_g(z)

	def configure_optimizers(self):
		"""See docs: https://pytorch-lightning.readthedocs.io/en/0.7.1/pl_examples.domain_templates.gan.html"""

		# Try different optimizers: AdaTune? RMSProp?
		opt_g = Adam(self.model_g.parameters(), lr=0.01)
		opt_d = Adam(self.model_d.parameters(), lr=0.02)

		sch_d = CosineAnnealingLR(opt_d, 10, 0.0001)

		return [opt_g, opt_d], [sch_d]

	def training_step(self, batch, batch_idx, optimizer_idx):
		[image_wm, image] = batch

		image_gen = self.forward(image_wm)

		fake_ab = torch.cat((image_wm, image_gen), 1)

		if optimizer_idx == 0:
			# First, G(A) should fake the discriminator
			pred_fake = self.model_d.forward(fake_ab)
			loss_g_gan = self.gan_loss(pred_fake, True)

			# Second, G(A) = B
			loss_g_l1 = F.l1_loss(image_gen, image) * 10  # parameterize this constant?

			loss = loss_g_gan + loss_g_l1

			assert not torch.isnan(loss), "loss is not a number!"

			return {"loss": loss}

		elif optimizer_idx == 1:
			# train with fake
			pred_fake = self.model_d.forward(fake_ab.detach())
			loss_d_fake = self.gan_loss(pred_fake, False)

			# train with real
			real_ab = torch.cat((image_wm, image), 1)
			pred_real = self.model_d.forward(real_ab)
			loss_d_real = self.gan_loss(pred_real, True)

			# Combined D loss
			loss = (loss_d_fake + loss_d_real) * 0.5

			assert not torch.isnan(loss), "loss is not a number!"

			return {"loss": loss}
		else:
			raise ValueError(f"Unexpected opt_id {optimizer_idx}")

	def validation_step(self, batch, batch_idx):
		[image_wm, image] = batch
		image_gen = self.forward(image_wm)
		mse = F.mse_loss(image_en, image)
        
        psnr = 20 * math.log10(255.0 / math.sqrt(mse.item()))
		ssim = compare_ssim(image_en.numpy(), image.numpy(), multichannel=True, data_range=255)

        return {'val_psnr': psnr, 'val_ssim':ssim}


	def validation_epoch_end(self, outputs):
		print(outputs)
		# avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
  #       tensorboard_logs = {'val_loss': avg_loss}
  #       return {'val_loss': avg_loss, 'log': tensorboard_logs}


class Generator(nn.Module):
	"""The generator module for the GAN"""

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
				 padding_type='reflect'):
		assert n_blocks > 0, "n_blocks must be greater than zero"

		super().__init__()

		use_bias = True

		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf

		self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
		self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
		self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)

		model = []
		for i in range(n_blocks):
			model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
							   use_bias=use_bias)]
		self.resblocks = nn.Sequential(*model)

		self.up1 = Up(ngf * 8, ngf * 2, norm_layer, use_bias)
		self.up2 = Up(ngf * 4, ngf, norm_layer, use_bias)

		self.outc = Outconv(ngf*2, output_nc)

	def forward(self, input):
		out = {}
		out['in'] = self.inc(input)
		out['d1'] = self.down1(out['in'])
		out['d2'] = self.down2(out['d1'])
		out['bottle'] = self.resblocks(out['d2'])
		out['u1'] = self.up1(torch.cat(out['bottle'], out['d2']))
		out['u2'] = self.up2(torch.cat(out['u1'], out['d1']))

		return self.outc(torch.cat(out['u2'], out['in']))


class Inconv(nn.Module):
	def __init__(self, in_ch, out_ch, norm_layer, use_bias):
		super(Inconv, self).__init__()
		self.inconv = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
					  bias=use_bias),
			norm_layer(out_ch),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.inconv(x)
		return x


class Down(nn.Module):
	def __init__(self, in_ch, out_ch, norm_layer, use_bias):
		super(Down, self).__init__()
		self.down = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3,
					  stride=2, padding=1, bias=use_bias),
			norm_layer(out_ch),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.down(x)
		return x


# Define a Resnet block
class ResBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim),
					   nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return nn.ReLU(True)(out)


class Up(nn.Module):
	def __init__(self, in_ch, out_ch, norm_layer, use_bias):
		super(Up, self).__init__()
		self.up = nn.Sequential(
			# nn.Upsample(scale_factor=2, mode='nearest'),
			# nn.Conv2d(in_ch, out_ch,
			#           kernel_size=3, stride=1,
			#           padding=1, bias=use_bias),
			nn.ConvTranspose2d(in_ch, out_ch,
							   kernel_size=3, stride=2,
							   padding=1, output_padding=1,
							   bias=use_bias),
			norm_layer(out_ch),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.up(x)
		return x


class Outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(Outconv, self).__init__()
		self.outconv = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.outconv(x)
		return x


class Discriminator(nn.Module):
	"""The discriminator module for the GAN"""

	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_bias=True):
		super().__init__()

		kw = 4
		padw = 1
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]

		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
						  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
					  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		if use_sigmoid:
			sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, x):
		x = x + torch.cuda.FloatTensor(*x.shape).normal_(0, 0.1) # add some noise to handicap the network

		return self.model(x)
