import DeviceHandling

from os import listdir
from os.path import join
from PIL import Image
from Settings import settings
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor


def prepareDataForTraining():
	datasets = createDatasetsAsDict()
	dataloaders = createDataloders(datasets)
	deviceDataloaders = createDeviceDataloaders(dataloaders)
	return deviceDataloaders	

def createDatasetsAsDict():
	wholeDataset = Custom_Pix2Pix_Dataset()
	numberOfData = len(wholeDataset)
	trainDataset, validationDataset, testDataset = random_split(wholeDataset, getSubsetSizesAsList(numberOfData))
	return {'train': trainDataset, 'validation': validationDataset, 'test': testDataset}

def createDataloders(datasets):
	trainDataLoader = DataLoader(datasets['train'], settings['batchSize'], shuffle=True)
	validationDataLoader = DataLoader(datasets['validation'], settings['batchSize'])
	testDataLoader = DataLoader(datasets['test'], settings['batchSize'])
	return {'train': trainDataLoader, 'validation': validationDataLoader, 'test': testDataLoader}

def createDeviceDataloaders(dataloaders):
	device = DeviceHandling.getCudaDeviceIfAvailable()
	trainDeviceDataLoader = DeviceHandling.DeviceDataLoader(dataloaders['train'], device)
	validationDeviceDataLoader = DeviceHandling.DeviceDataLoader(dataloaders['validation'], device)
	testDeviceDataLoader = DeviceHandling.DeviceDataLoader(dataloaders['test'], device)
	return {'train': trainDeviceDataLoader, 'validation': validationDeviceDataLoader, 'test': testDeviceDataLoader}

def getSubsetSizesAsList(fullDatasetSize):
	validationSetSize = round(fullDatasetSize * settings['validationSetProportion'])
	testSetSize = round(fullDatasetSize * settings['testSetProportion'])
	trainSetSize = fullDatasetSize - validationSetSize - testSetSize
	return [trainSetSize, validationSetSize, testSetSize]


class Custom_Pix2Pix_Dataset(Dataset):
	def __init__(self, transform=ToTensor()):
		self.dataFolder = settings['dataFolder']
		self.inputPath = join(self.dataFolder, settings['inputSubfolder'])
		self.targetPath = join(self.dataFolder, settings['targetSubfolder'])
		self.transform = transform
		self.numberOfFiles = len(listdir(self.inputPath))
		
	def __len__(self):
		return self.numberOfFiles

	def __getitem__(self, index):
		inputImagePath = join(self.inputPath, f'Input ({index+1}).png')
		inputImage = Image.open(inputImagePath)
		targetImagePath = join(self.targetPath, f'Target ({index+1}).png')
		targetImage = Image.open(targetImagePath)

		if self.transform:
			inputImage = self.transform(inputImage)
			targetImage = self.transform(targetImage)

		return inputImage, targetImage