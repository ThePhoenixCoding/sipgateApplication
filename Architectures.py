import torch
import torch.nn as nn

class U_Net(nn.Module):
	'''Building full U-Net model from encoder and decoder blocks'''

	# Create blocks of model
	def __init__(self):
		super().__init__()

		# Encoder
		self.encode1 = encoder_block(input_channels=4, output_channels=128)
		self.encode2 = encoder_block(input_channels=128, output_channels=256)
		self.encode3 = encoder_block(input_channels=256, output_channels=512)
		self.encode4 = encoder_block(input_channels=512, output_channels=1024)

		# Decoder
		self.decode1 = decoder_block(input_channels=1024, intermediate_channels=2048, output_channels=1024)
		self.decode2 = decoder_block(input_channels=2048, intermediate_channels=1024, output_channels=512)
		self.decode3 = decoder_block(input_channels=1024, intermediate_channels=512, output_channels=256)
		self.decode4 = decoder_block(input_channels=512, intermediate_channels=256, output_channels=128)
		self.decode5 = decoder_block(input_channels=256, intermediate_channels=128, output_channels=128, last_block_channels=4) # output RGB-Image

	# Create connections of model
	def forward(self, input3D):
		# Encode
		skip1, processed1 = self.encode1(input3D)
		skip2, processed2 = self.encode2(processed1)
		skip3, processed3 = self.encode3(processed2)
		skip4, processed4 = self.encode4(processed3)

		# Decode
		processed5 = self.decode1(processed4)
		processed6 = self.decode2(self.concatenateFeatureMaps(processed5, skip4))
		processed7 = self.decode3(self.concatenateFeatureMaps(processed6, skip3))
		processed8 = self.decode4(self.concatenateFeatureMaps(processed7, skip2))
		processed9 = self.decode5(self.concatenateFeatureMaps(processed8, skip1))

		return processed9

	def concatenateFeatureMaps(self, data1, data2):
		return torch.cat([data1, data2], axis=1)

class encoder_block(nn.Module):
	'''Represents one horizontal block of U_nets left half incl. pooling'''
	def __init__(self, input_channels, output_channels):
		super().__init__()

		self.convolution1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
		self.convolution2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

		self.batchNormalizationFunction = nn.BatchNorm2d(output_channels)
		self.activationFunction = nn.ReLU()
		self.poolingFunction = nn.MaxPool2d((2, 2))

	def forward(self, input):
		out = self.convolution1(input)
		out = self.batchNormalizationFunction(out)
		out = self.activationFunction(out)

		out = self.convolution2(out)
		out = self.batchNormalizationFunction(out)
		out = self.activationFunction(out)
		
		pooled_out = self.poolingFunction(out)		
		return out, pooled_out

class decoder_block(nn.Module):
	'''Represents one horizontal block of U_nets right half incl. up-convolution'''
	def __init__(self, input_channels, intermediate_channels, output_channels, last_block_channels=0):
		super().__init__()

		self.convolution1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1)
		self.convolution2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1)

		self.batchNormalizationFunction = nn.BatchNorm2d(intermediate_channels)
		self.activationFunction = nn.ReLU()
		if last_block_channels == 0:
			self.up_convolution = nn.ConvTranspose2d(intermediate_channels, output_channels, kernel_size=2, stride=2)
		else:
			self.up_convolution = nn.Conv2d(output_channels, last_block_channels, kernel_size=1)

	def forward(self, input):
		out = self.convolution1(input)
		out = self.batchNormalizationFunction(out)
		out = self.activationFunction(out)

		out = self.convolution2(out)
		out = self.batchNormalizationFunction(out)
		out = self.activationFunction(out)

		out = self.up_convolution(out)
		return out