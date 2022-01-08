import Architectures
import Dataprocessing
import DeviceHandling
import Metrics
import Output
import time
import torch.nn as nn
import Utilities

from Settings import settings

def trainModelIfNoTrainedModelAvailable():
	model = Utilities.loadTrainedModelOrNone()
	if model == None:
		data = Dataprocessing.prepareDataForTraining()
		model = Architectures.U_Net()
		DeviceHandling.moveDataToDevice(model, DeviceHandling.getCudaDeviceIfAvailable())
		validationHistory = trainAndTestModel(model, data)
		Output.saveTrainingDiagram(validationHistory)
		Utilities.saveTrainedModel(model)
	else:
		print("Pretrained model found. Skipping Training.")
	return model

def trainAndTestModel(model, data):
	print("Training model:")
	startTime = time.perf_counter()
	history = trainModel(model, data)
	testModel(model, data['test'])
	elapsedTime = time.perf_counter()-startTime
	print('Trained for : ' + str(round(elapsedTime,2)) + "s")
	return history

def trainModel(model, data):
	optimizer = settings['optimizer'](model.parameters(), lr=settings['learningRate'], weight_decay = settings['weightDecay'])
	history = {'losses': [], 'ssims': []}
	evaluateModelAndAppendToHistory(model, data['validation'], history)
	
	 # Train & validate for defined numbers of epochs
	for epoch in range(settings['epochs']):
		# Training Phase: Generate predictions, calculate es and adapt the model
		for batch in data['train']:
			trainingStep(model, batch, optimizer)

		#Validation phase: Calculate metrics with validation data
		evaluateModelAndAppendToHistory(model, data['validation'], history)
	return history

def testModel(model, dataloader):
	print("\nTesting model:")
	metricsFromTesting = calculateMetrics(model, dataloader)
	printMetrics(metricsFromTesting)

def evaluateModelAndAppendToHistory(model, dataloader, history):
		metrics = calculateMetrics(model, dataloader)
		history['losses'].append(metrics['loss'])
		history['ssims'].append(metrics['ssim'])		
		if settings['verboseTraining']:
			print(f"Result for epoch {len(history['losses'])-1}/{settings['epochs']}: Mean loss: {metrics['loss']}, Mean MAE: {metrics['mae']}, Mean PSNR: {metrics['psnr']}, Mean SSIM: {metrics['ssim']}")

def calculateMetrics(model, dataloader):
	predictionsAndTargets = calculatePredictions(model, dataloader)
	metrics = Metrics.calculateMeanMetricsFromPredictions(predictionsAndTargets)
	return metrics

def trainingStep(model, batch, optimizer):
	inputs, targets = batch
	predictions = model(inputs)
	loss = Metrics.getLossFunction()(predictions, targets)
	loss.backward()
	if settings['gradientClipping']: # If maximum gradient value is set then cut off gradients greater than that
		nn.utils.clip_grad_value_(model.parameters(), settings['gradientClipping'])
	optimizer.step()
	optimizer.zero_grad()

def printMetrics(metrics):
	print('Mean test loss (MSE):', metrics['loss'])
	print('Mean test MAE:', metrics['mae'])
	print('Mean test PSNR:', metrics['psnr'])
	print('Mean test SSIM:', metrics['ssim'])

def calculatePredictions(model, dataloader):
	output = []
	for (inputs, targets) in dataloader:
			predictions = model(inputs).detach()
			output.append((predictions, targets))
	return output