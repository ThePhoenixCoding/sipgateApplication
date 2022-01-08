import Output
import Training
import Utilities

Utilities.setOutputToConsoleAndLogfile()
model = Training.trainModelIfNoTrainedModelAvailable()
Output.generateAndSavePredictionsAsImages(model)