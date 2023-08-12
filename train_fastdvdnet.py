"""
Trains a FastDVDnet model.

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import FastDVDnet
from dataset import ValDataset
from dataloaders import train_dali_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint
from PIL import Image
import imagehash as imghash
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gc

def main(**args):
	r"""Performs the main training loop
	"""

	# Load dataset
	print('> Loading datasets ...')
	dataset_val = ValDataset(valsetdir_noisy=args['valset_dir_noisy'], \
							 valsetdir_denoised=args['valset_dir_denoised'], \
							 valsetdir_original=args['valset_dir_original'], \
							 gray_mode=False)
	loader_train = train_dali_loader(batch_size=args['batch_size'],\
									noisy_file_root=args['trainset_dir_noisy'], \
									denoised_file_root=args['trainset_dir_denoised'], \
									original_file_root=args['trainset_dir_original'], \
									sequence_length=args['temp_patch_size'],\
									crop_size=args['patch_size'],\
									epoch_size=args['max_number_patches'],\
									temp_stride=3)

	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = FastDVDnet()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('\nlearning rate %f' % current_lr)

		# train
		eventHorizonCrossed = False

		for i, data in enumerate(loader_train, 0):
			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			imgo_train, imgn_train, imgd_train, gt_train, gt_n, gt_d = normalize_augment(data[0]['data_original'], \
																						 data[0]['data_noisy'], \
																						 data[0]['data_denoised'], \
																						 ctrl_fr_idx)
			# imgo_train, gt_train = normalize_augment(data[0]['data_original'], ctrl_fr_idx)
			# imgn_train, gt_n = normalize_augment(data[0]['data_noisy'], ctrl_fr_idx)
			# imgd_train, gt_d = normalize_augment(data[0]['data_denoised'], ctrl_fr_idx)
			N, _, H, W = imgn_train.size()

			# dd = data[0]['data_denoised']
			# dd = dd.view(dd.size()[0], -1, dd.size()[-2], dd.size()[-1]) / 255.
			# dd = dd[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
			#
			# dn = data[0]['data_noisy']
			# dn = dn.view(dn.size()[0], -1, dn.size()[-2], dn.size()[-1]) / 255.
			# dn = dn[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
			#
			# do = data[0]['data_original']
			# do = do.view(do.size()[0], -1, do.size()[-2], do.size()[-1]) / 255.
			# do = do[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]

			# if(do.size() != dd.size()):
			# 	print("Sizes differ {} to {}".format(dd.size(), do.size()))
			#
			# if(areImagesDesincronized(do[0], dd[0])):
			# 	eventHorizonCrossed = True
			# 	print("AAAAAAAAAAAAAAAAA")

			# if(eventHorizonCrossed):
				# showImage(do[0], "1 {}. Original from pipeline: ".format(training_params['step']))
				# showImage(do[1], "2 {}. Original from pipeline: ".format(training_params['step']))
				# showImage(do[2], "3 {}. Original from pipeline: ".format(training_params['step']))
				# showImage(dd[0], "1 {}. Denoised from pipeline: ".format(training_params['step']))
				# showImage(dd[1], "2 {}. Denoised from pipeline: ".format(training_params['step']))
				# showImage(dd[2], "3 {}. Denoised from pipeline: ".format(training_params['step']))

			# if training_params['step'] % 200 == 0:
				# showImage(do[0], "{}. Original from pipeline: ".format(training_params['step']/200))
				# showImage(dd[0], "{}. Denoised from pipeline: ".format(training_params['step']/200))
				# showImage(dn[0], "{}. Noisy from pipeline: ".format(training_params['step'] / 200))
				# showImage(dd[1], "2. Denoised from pipeline: ")
				# showImage(dd[2], "3. Denoised from pipeline: ")
				# showImage(dn[1], "2. Noisy from pipeline: ")
				# showImage(dn[2], "3. Noisy from pipeline: ")
				# showImage(gt_train[0], "GT0")
				# showImage(gt_n[0], "GTN")
				# showImage(gt_d[0], "GTD")

			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			imgd_train = imgd_train.cuda(non_blocking=True)

			# Evaluate model and optimize it
			out_train = model(imgn_train, imgd_train)
			# if training_params['step'] > 1000:
			# 	showImage(out_train[0], "OUT0")

			# Compute loss
			loss = criterion(gt_train, out_train) / (N*2)
			loss.backward()
			optimizer.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

			# Validation and log images
			# validate_and_log(
			# 	model_temp=model, \
			# 	dataset_val=dataset_val, \
			# 	valnoisestd=args['val_noiseL'], \
			# 	temp_psz=args['temp_patch_size'], \
			# 	writer=writer, \
			# 	epoch=epoch, \
			# 	lr=current_lr, \
			# 	logger=logger, \
			# 	trainimg=imgo_train
			# )

		# Call to model.eval() to correctly set the BN layers before inference
		model.eval()

		# Validation and log images
		validate_and_log(
						model_temp=model, \
						dataset_val=dataset_val, \
						valnoisestd=args['val_noiseL'], \
						temp_psz=args['temp_patch_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger, \
						trainimg=imgo_train
						)

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)



	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

def showImage(img, t):
	image = img.cpu()  # Extract the image at index i

	# Assuming the tensor is in [0, 1] range, you can convert it to [0, 255] range
	image = (image * 255).byte()

	# Convert tensor to numpy array and rearrange dimensions from [C, H, W] to [H, W, C]
	image = image.permute(1, 2, 0).numpy()

	# Display the image
	plt.imshow(image)
	plt.title(f"Image {t}")
	plt.show()

def areImagesDesincronized(img1, img2):
	image1 = img1.cpu()  # Extract the image at index i
	image1 = (image1 * 255).byte()
	image1 = image1.permute(1, 2, 0).numpy()
	i1 = Image.fromarray(image1)

	image2 = img2.cpu()  # Extract the image at index i
	image2 = (image2 * 255).byte()
	image2 = image2.permute(1, 2, 0).numpy()
	i2 = Image.fromarray(image2)

	hash0 = imghash.average_hash(i1)
	hash1 = imghash.average_hash(i2)

	cutoff = 20  # maximum bits that could be different between the hashes.
	diff = hash0 - hash1

	if diff >= cutoff:
		#plt.imshow(i1)
		#plt.show()
		#plt.imshow(i2)
		#plt.show()
		return True
	return False

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=64, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir_noisy", type=str, default=None, \
					 help='path of noisy trainset')
	parser.add_argument("--trainset_dir_denoised", type=str, default=None, \
						help='path of denoised trainset')
	parser.add_argument("--trainset_dir_original", type=str, default=None, \
						help='path of original trainset')
	parser.add_argument("--valset_dir_noisy", type=str, default=None, \
						 help='path of validation set')
	parser.add_argument("--valset_dir_denoised", type=str, default=None, \
						help='path of validation set')
	parser.add_argument("--valset_dir_original", type=str, default=None, \
						help='path of validation set')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
