from keras.models import *
from keras.layers import *
from keras import backend as K
from keras import regularizers
import tensorflow as tf


"""
	超参数：
		OPT = 'adam'
		LOSS = 'binary_crossentropy'
		L2_ALPHA = 0.02
		dropout_ALPHA = 0.25
		BATCH_SIZE = 256
		EPOCHS = 50
		IMAGE_SIZE = (150, 150, 3)
"""


def model(input_shape, OPT, LOSS, L2_ALPHA, dropout_ALPHA):
	def auc(y_true, y_pred):
		def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
			y_pred = K.cast(y_pred >= threshold, 'float32')
			N = K.sum(1 - y_true)
			FP = K.sum(y_pred - y_pred * y_true)

			return FP / N

		def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
			y_pred = K.cast(y_pred >= threshold, 'float32')
			P = K.sum(y_true)
			TP = K.sum(y_pred * y_true)

			return TP / P

		ptas = tf.stack([binary_PTA(y_true,y_pred,k) \
				for k in np.linspace(0, 1, 1000)], axis=0)
		pfas = tf.stack([binary_PFA(y_true,y_pred,k) \
				for k in np.linspace(0, 1, 1000)], axis=0)
		pfas = tf.concat([tf.ones((1,)) ,pfas], axis=0)

		binSizes = -(pfas[1:] - pfas[:-1])
		s = ptas * binSizes

		return K.sum(s, axis=0)

	inputs = Input(shape=input_shape, name="inputs")

	conv1 = Conv2D(32,
				   (3, 3),
				   name="conv1",
				   kernel_regularizer=regularizers.l2(L2_ALPHA),
				   kernel_initializer="RandomNormal")(inputs)
	relu1 = Activation('relu', name="relu1")(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(relu1)
	
	conv2 = Conv2D(64,
				   (3, 3),
				   name = "conv2",
				   kernel_regularizer=regularizers.l2(L2_ALPHA),
				   kernel_initializer="RandomNormal")(pool1)
	relu2 = Activation('relu', name="relu2")(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(relu2)
	
	conv3 = Conv2D(64,
				   (3, 3),
				   name = "conv3",
				   padding='same',
				   kernel_regularizer=regularizers.l2(L2_ALPHA),
				   kernel_initializer="RandomNormal")(pool2)
	relu3 = Activation('relu', name="relu3")(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(relu3)
	
	conv4 = Conv2D(128,
				   (3, 3),
				   name = "conv4",
				   kernel_regularizer=regularizers.l2(L2_ALPHA),
				   kernel_initializer="RandomNormal")(pool3)
	relu4 = Activation('relu', name="relu4")(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(relu4)
	
	x = Flatten()(pool4)
	x = Dropout(dropout_ALPHA)(x)
	
	x = Dense(1024, activation='relu', kernel_initializer="RandomNormal")(x)
	x = Dropout(dropout_ALPHA)(x)
	x = Dense(256, activation='relu', kernel_initializer="RandomNormal")(x)
	out = Dense(1, activation="sigmoid", kernel_initializer="RandomNormal")(x)

	model = Model(inputs=inputs, outputs=out)
	model.compile(optimizer=OPT, loss=LOSS, metrics=[auc])

	return model
