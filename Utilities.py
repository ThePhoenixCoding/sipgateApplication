import Architectures
import DeviceHandling
import sys
import torch

from datetime import datetime
from os.path import isfile

def getDateAndTime():
	dateTimeStr = str(datetime.now())
	dateTimeStrWithoutMS = dateTimeStr[:-7]
	dateTimeFilenameLegal = dateTimeStrWithoutMS.replace(":", "-")
	return dateTimeFilenameLegal

def loadTrainedModelOrNone():
	if isfile('trainedUNet.pth'):
		model = Architectures.U_Net()
		model.load_state_dict(torch.load('trainedUNet.pth'))
		DeviceHandling.moveDataToDevice(model, DeviceHandling.getCudaDeviceIfAvailable())
		return model
	return None

def saveTrainedModel(model):
	if isinstance(model, Architectures.U_Net):
		torch.save(model.state_dict(), 'trainedUNet.pth')
	else:
		print("Could not save trained model. Architecture unknown.")


class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		filename = "log " + getDateAndTime() + ".txt"
		self.logFile = open(filename, "w")

	def write(self, message):
		self.terminal.write(message)
		self.logFile.write(message)  

	def flush(self):
		self.logFile.flush()

def setOutputToConsoleAndLogfile():
	sys.stdout = Logger()