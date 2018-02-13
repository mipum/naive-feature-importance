import numpy as np
from keras.models import Model

def feature_off_generator( X, axis):
	'''
	Generator yielding modified test input with subsequent features switched off (seroed) across all samples
	Parameters
		X : inputs tensor
		axis : axis of X containing the features to be evaluated
	Returns
		copy of X with one feature a time zeroed off
	'''	
	# prepare a slicer template
	slcr = [slice(None)] * X.ndim

	# traverse the given axis
	for f in range(X.shape[axis]):
		# adjust slicer
		slcr[axis] = f

		# prepare filter tensor
		fltr = np.ones(X.shape)
		fltr[slcr] = 0

		# return copy of X with filter applied
		yield np.multiply( X, fltr)

# create, compile and fit a model 'model'

# get list of model accuracies per each feature zeroed in test data X		
importancies = [model.evaluate( X_prime, y)[1] for X_prime in feature_off_generator( X, axis)]
