import torch

from Settings import settings

# Code for using the GPU adapted from:
#https://jovian.ai/aakashns/04-feedforward-nn/v/29#C65


def moveDataToDevice(data, device):
	if isNestedDatastructure(data):
		return [moveDataToDevice(element, device) for element in data]
	return data.to(device, non_blocking=True)

def isNestedDatastructure(data):
	return isinstance(data, (list, tuple))

def getCudaDeviceIfAvailable():
	if torch.cuda.is_available() and not settings['forceTrainingOnCPU']:
		return torch.device('cuda')
	else:
		return torch.device('cpu')


class DeviceDataLoader():
	'''Dataloader that automatically moves batch to device before giving data'''
	def __init__(self, originalDataloader, device):
		self.originalDataloader = originalDataloader
		self.device = device
		
	def __iter__(self):
		for batch in self.originalDataloader: 
			yield moveDataToDevice(batch, self.device)

	def __len__(self):
		return len(self.originalDataloader)