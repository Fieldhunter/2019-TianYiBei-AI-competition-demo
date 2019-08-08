import os
from keras.preprocessing import image
import numpy as np
import glob
import cv2
from keras import backend as K
import random
import tensorflow as tf
import tensorflow.gfile as gfile
import pickle
from model import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 超参数
OPT = 'adam'
LOSS = 'binary_crossentropy'
dropout_ALPHA = 0.5
L2_ALPHA = 0.02
BATCH_SIZE = 256
EPOCHS = 45
IMAGE_SIZE = (224, 224)

# 参数
DATA_PATH = "data/Trainset/"
MASK_PATH = "data/mask/"
MODEL_DIR = './model/first/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/first/'
HISTORY_FORMAT = '.history'
filename_str = "{}captcha_{}_{}_bs_{}_epochs_{}{}"
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, \
				str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, \
				str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)


# 核对通道位置
def fit_keras_channels(batch, rows=IMAGE_SIZE[0], cols=IMAGE_SIZE[1]):
	if K.image_data_format() == 'channels_first':
		batch = batch.reshape(batch.shape[0], 3, rows, cols)
		input_shape = (3, rows, cols)
	else:
		batch = batch.reshape(batch.shape[0], rows, cols, 3)
		input_shape = (rows, cols, 3)

	return batch, input_shape


def load_data():
	# 对mask图案的变换
	def mask_process(make_img):
		# 随机平移
		process_image = image.random_shift(make_img,
										   0.3,
										   0.3,
										   row_axis=0,
										   col_axis=1,
										   channel_axis=2,
										   fill_mode='constant',
										   cval=0)

		# 随机旋转
		process_image = image.random_rotation(process_image,
											  180,
											  row_axis=0,
											  col_axis=1,
											  channel_axis=2,
											  fill_mode='constant',
											  cval=0)

		return process_image

	# 对合成后图像的变换
	def original_process(original_img):
		# 随机旋转
		process_image = image.random_rotation(original_img,
											  180,
											  row_axis=0,
											  col_axis=1,
											  channel_axis=2,
											  fill_mode='constant',
											  cval=0)
			
		# 随机投影变换
		h, w = IMAGE_SIZE[0], IMAGE_SIZE[1]

		ratio = np.random.normal(0.125, 0.075)
		if ratio > 0.2:
			ratio = 0.2
		elif ratio < 0.05:
			ratio = 0.05
		pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
		pts2 = np.float32([[0, 0], [w, ratio*h], [0, h], [w, (1-ratio)*h]])

		M = cv2.getPerspectiveTransform(pts1, pts2)
		process_image = cv2.warpPerspective(process_image, M, (w, h))

		# 随机平移
		process_image = image.random_shift(process_image,
										   0.3,
										   0.3,
										   row_axis=0,
										   col_axis=1,
										   channel_axis=2,
										   fill_mode='constant',
										   cval=0)

		# 随机拉伸
		ratio = np.random.normal(35,10)
		if ratio > 45:
			ratio = 45
		elif ratio < 25:
			ratio = 25
		process_image = image.random_shear(process_image,
										   ratio,
										   row_axis=0,
										   col_axis=1,
										   channel_axis=2)

		return process_image


	# 掩膜合成
	def make_mask(original_img, mask_img):
		rows, cols, channels = mask_img.shape
		roi = original_img[0:rows, 0:cols] 
		mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

		ret, mask = cv2.threshold(mask_img_gray, 20, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		img1_bg = cv2.bitwise_and(mask_img, mask_img, mask=mask)
		img2_fg = cv2.bitwise_and(roi, roi, mask=mask_inv)
		dst = cv2.add(img1_bg, img2_fg)
		original_img[0:rows, 0:cols] = dst
		
		return original_img


	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	mask_list = glob.glob(MASK_PATH + '*.jpg')
	train_or_test = 0
	
	for filename in glob.glob(DATA_PATH + '*.jpg'):
		try:
			# 先进行mask图案的处理，之后合成掩膜，再对合成后的图案进行处理
			original_img = image.load_img(filename, target_size=IMAGE_SIZE)
			original_img = np.uint8(image.img_to_array(original_img))
			mask_img = image.load_img(random.choice(mask_list), target_size=IMAGE_SIZE)
			mask_img = image.img_to_array(mask_img)

			mask_img = np.uint8(mask_process(mask_img))
			original_img = make_mask(original_img, mask_img)
			original_img = original_process(np.float32(original_img))

			# 以交叉的方式选取1000张图片为测试集
			if train_or_test == 1 and len(X_test) < 1000:
				X_test.append(original_img)
				Y_test.append(filename.lstrip(DATA_PATH)[0])
				train_or_test = 0
			else:
				X_train.append(original_img)
				Y_train.append(filename.lstrip(DATA_PATH)[0])
				train_or_test = 1
		except:
			pass

	return X_train, Y_train, X_test, Y_test


# 进行数值归一化和格式设置
def process_data(X_train, Y_train, X_test, Y_test):
	X_train = np.array(X_train, dtype=np.float32)
	X_train = X_train / 255
	X_test = np.array(X_test, dtype=np.float32)
	X_test = X_test / 255

	X_train, input_shape = fit_keras_channels(X_train)
	X_test, _ = fit_keras_channels(X_test)

	Y_train = np.asarray(Y_train).reshape(len(Y_train),1)
	Y_test = np.asarray(Y_test).reshape(len(Y_test),1)

	return X_train, Y_train, X_test, Y_test, input_shape


# 保存模型和历史记录
def save_model(model, history):
	if not gfile.Exists(MODEL_DIR):
		gfile.MakeDirs(MODEL_DIR)

	model.save(MODEL_FILE)

	if gfile.Exists(HISTORY_DIR) == False:
		gfile.MakeDirs(HISTORY_DIR)

	with open(HISTORY_FILE, 'wb') as f:
		pickle.dump(history.history, f)


if __name__ == "__main__":
	X_train, Y_train, X_test, Y_test = load_data()
	X_train, Y_train, X_test, Y_test, input_shape = \
		process_data(X_train, Y_train, X_test, Y_test)
	
	model = model(input_shape, OPT, LOSS, L2_ALPHA, dropout_ALPHA)
	history = model.fit(X_train,
						Y_train,
						batch_size=BATCH_SIZE,
						epochs=EPOCHS,
						verbose=2,
						validation_data=(X_test, Y_test))

	save_model(model, history)
