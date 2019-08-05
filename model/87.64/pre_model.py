from keras.models import *
from keras.layers import *
from keras import backend as K
from keras import regularizers
import tensorflow as tf


"""
	超参数:
		OPT = 'adam'
		LOSS = 'binary_crossentropy'
		dropout_ALPHA = 0.5
		L2_ALPHA = 0.02
		BATCH_SIZE = 256
		EPOCHS = 45
		IMAGE_SIZE = (150, 150)
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

	def res_block(x, kernel_size, padding="same"):
		con1 = Conv2D(kernel_size,
					  (3, 3),
					  kernel_regularizer=regularizers.l2(L2_ALPHA),
					  kernel_initializer="RandomNormal",
					  padding=padding)(x)
		rel1 = Activation('relu')(con1)

		con2 = Conv2D(kernel_size,
					  (3, 3),
					  kernel_regularizer=regularizers.l2(L2_ALPHA),
					  kernel_initializer="RandomNormal",
					  padding=padding)(rel1)

		return add([x, con2])


	inputs = Input(shape=input_shape)

	conv1 = Conv2D(32,
				   (3, 3),
				   kernel_regularizer=regularizers.l2(L2_ALPHA))(inputs)
	relu1 = Activation('relu')(conv1) 
	
	res1 = res_block(relu1, 32)
	relu2 = Activation('relu')(res1)

	pool1 = MaxPooling2D(pool_size=(2, 2))(relu2)

	conv2 = Conv2D(64,
				   (3, 3),
				   kernel_regularizer=regularizers.l2(L2_ALPHA))(pool1)
	relu3 = Activation('relu')(conv2)
	
	res2 = res_block(relu3, 64)
	relu4 = Activation('relu')(res2)

	pool2 = MaxPooling2D(pool_size=(2, 2))(relu4)
	
	conv3 = Conv2D(128,
				   (3, 3),
				   padding='same',
				   kernel_regularizer=regularizers.l2(L2_ALPHA))(pool2)
	relu5 = Activation('relu')(conv3)

	res3 = res_block(relu5, 128)
	relu6 = Activation('relu')(res3)

	pool3 = MaxPooling2D(pool_size=(2, 2))(relu6)

	conv4 = Conv2D(256,
				   (3, 3),
				   kernel_regularizer=regularizers.l2(L2_ALPHA))(pool3)
	relu7 = Activation('relu')(conv4)

	res4 = res_block(relu7, 256)
	relu8 = Activation('relu')(res4)

	pool4 = MaxPooling2D(pool_size=(2, 2))(relu8)

	x = Flatten()(pool4)
	
	x = Dropout(dropout_ALPHA)(x)
	x = Dense(2048, activation='relu')(x)
	x = Dropout(dropout_ALPHA)(x)
	x = Dense(2048, activation='relu')(x)

	out = Dense(1, activation="sigmoid")(x)

	model = Model(inputs=inputs, outputs=out)
	model.compile(optimizer=OPT, loss=LOSS, metrics=[auc])

	return model
