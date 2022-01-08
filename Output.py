import Dataprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch
import Utilities

from os.path import join
from Settings import settings

def generateAndSavePredictionsAsImages(model):
	outputPath = join(settings['dataFolder'], settings['destinationSubfolder'])
	print("Generating predictions.")
	data = Dataprocessing.prepareDataForTraining()
	dataWithPredictions = createPredictionsForWholeData(model, data)
	print(f"Saving predictions in {outputPath}")
	savePredictedDataAsImage(dataWithPredictions, outputPath)

def createPredictionsForWholeData(model, wholeData):
	results = []
	for set in wholeData:
		for (inputBatch, targetBatch) in wholeData[set]:
			predictionBatch = model(inputBatch).detach()
			for i in range(len(inputBatch)): # get images from batches
				results.append((inputBatch[i].cpu(), targetBatch[i].cpu(), predictionBatch[i].cpu()))
	return results

def savePredictedDataAsImage(dataWithPredictions, outputPath):
	numberOfPictures = len(dataWithPredictions)
	for i in range(numberOfPictures):
		images = prepareImagesForSaving(dataWithPredictions[i])
		filename = join(outputPath, "Prediction (" + str(i+1) + ").png")
		saveFigure(images, filename)

def prepareImagesForSaving(images):
	permutedTuple = moveColorChannelToLastDimension(images)
	normalizedPrediction = normalizeImage(permutedTuple[2])
	return permutedTuple[0], permutedTuple[1], normalizedPrediction

def saveFigure(images, filename):
		_, ax = plt.subplots(1,3)
		ax[0].set_title("Input")
		ax[0].imshow(images[0])
		ax[1].set_title("Target")
		ax[1].imshow(images[1])
		ax[2].set_title("Prediction")
		ax[2].imshow(images[2])
		plt.savefig(filename, dpi=300)
		plt.close()


def moveColorChannelToLastDimension(images):
	permutedInput = torch.permute(images[0], (1,2,0))
	permutedTarget = torch.permute(images[1], (1,2,0))
	permutedPrediction = torch.permute(images[2], (1,2,0))
	return permutedInput, permutedTarget, permutedPrediction

def normalizeImage(image):
	image -= torch.min(image)
	image /= torch.max(image)
	return image

def saveTrainingDiagram(validationHistory):
	epochs = settings['epochs']
	_, ax = plt.subplots() 

	x = np.arange(0, epochs+1, 1)
	ax.plot(x, validationHistory['losses'], label="Loss")
	ax.plot(x, validationHistory['ssims'], label="SSIM")

	configureAxes(ax, epochs+1)

	ax.grid()
	plt.legend()
	plt.savefig("Losses " + Utilities.getDateAndTime() + ".png", dpi=150)
	plt.close()

def configureAxes(ax, datapoints):
	ax.set(xlabel='Epochs', ylabel='', title='Validation History')
	ax.set_xticks([x for x in range(datapoints)])
	ax.set_ylim([0.01, 1.0])
	ax.set_yscale('function', functions=(scalingForward, scalingInverse))
	ax.set_yticks([0.025, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])

# Define scaling of y-axis in plotting:
def scalingForward(x):
		return x**(1/4)
def scalingInverse(x):
		return x**4