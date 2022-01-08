import numpy as np
import torch
import torch.nn as nn

from skimage.metrics import structural_similarity as ssim_f

mae_f = nn.L1Loss()
mse_f = nn.MSELoss()
loss_fn = mse_f

def getLossFunction():
	return loss_fn

def calculateMeanMetricsFromPredictions(batch_loader):
	metrics = calculateMetricsFromPredictions(batch_loader)
	meanMetrics = calculateMeansForMetricsDict(metrics)
	return meanMetrics

def calculateMetricsFromPredictions(batch_loader):
	metrics = {'losses': [], 'maes': [], 'psnrs': [], 'ssims': []}

	for (predictions, targets) in batch_loader:
		loss = calculateLossItem(predictions, targets)
		metrics['losses'].append(loss)

		psnr = calculatePSNR(predictions, targets)
		metrics['psnrs'].append(psnr)
		
		mae = calculateMAEItem(predictions, targets)
		metrics['maes'].append(mae)

		ssim = calculateSSIM(predictions, targets)
		metrics['ssims'].append(ssim)

	return metrics

def calculateMeansForMetricsDict(dict):
	meanMetrics = {}
	meanMetrics['loss'] = np.mean(dict['losses'])
	meanMetrics['psnr'] = np.mean(dict['psnrs'])
	meanMetrics['mae'] = np.mean(dict['maes'])
	meanMetrics['ssim'] = np.mean(dict['ssims'])
	return meanMetrics

def calculateLossItem(predictions, targets):
	return loss_fn(predictions, targets).item()

def calculatePSNR(predictions, targets):
	return psnr_f(predictions, targets)

def calculateMAEItem(predictions, targets):
	return mae_f(predictions, targets).item()

def calculateSSIM(predictions, targets):
		predictions = prepareBatchesForSkimageSSIM(predictions)
		targets = prepareBatchesForSkimageSSIM(targets)
		ssim = ssim_f(predictions, targets, multichannel=True)
		return max(ssim, 0)

def prepareBatchesForSkimageSSIM(batch):
	'''The Skimage SSIM implementation expects the colorchannel/batch count to be the last dimension of a numpy array. Pytorch stores batch count as first in first dimension in tensor arrays'''
	batch = torch.flatten(batch, start_dim=0, end_dim=1) # remove dummy dimension from pictures
	batch = torch.permute(batch, (1,2,0))
	batch = batch.cpu().numpy()
	return batch

# Maths by https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr_f (prediction, target):
	mse = mse_f(prediction, target)
	return (20 * torch.log10(torch.max(target)) - 10 * torch.log10(mse)).item()