from keras.models import *
from keras.layers import *
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import tensorflow as tf


"""
	超参数：
		OPT = 'adam'
		LOSS = 'binary_crossentropy'
		dropout_ALPHA = 0.45
		BATCH_SIZE = 256
		EPOCHS = 45
		IMAGE_SIZE = (299, 299)
"""


def model(input_shape, OPT, LOSS, dropout_ALPHA):
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

	# 构建不带分类器的预训练模型
	base_model = InceptionV3(weights='imagenet', include_top=False)

	# 添加全局平均池化层
	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	# 添加全连接层
	x = Dropout(dropout_ALPHA)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(dropout_ALPHA)(x)
	x = Dense(256, activation='relu')(x)

	# 添加一个分类器
	predictions = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	# 锁住inceptionv3
	for layer in base_model.layers:
		layer.trainable = False

	# 编译模型（一定要在锁层以后操作）
	model.compile(optimizer=OPT, loss=LOSS, metrics=[auc])

	return model
