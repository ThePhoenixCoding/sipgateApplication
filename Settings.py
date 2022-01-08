import torch.optim

settings = {
	# Names of data folders
	'dataFolder': 'Dataset', 
	'inputSubfolder': 'Input', 
	'targetSubfolder': 'Target',
	'destinationSubfolder': 'Generated' ,
	
	# Training parameters
	'forceTrainingOnCPU': False,
	'verboseTraining': True,
	'validationSetProportion': 0.15,
	'testSetProportion': 0.15,
	'batchSize': 4,
	'optimizer': torch.optim.Adam,
	'epochs': 20,
	'learningRate': 1e-4,
	'gradientClipping': 0,
	'weightDecay': 1e-4
	}