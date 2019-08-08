import glob
from keras.preprocessing import image
import numpy as np
import pandas as pd

# 模型以及参数位置
MODEL_NAME = "captcha_adam_binary_crossentropy_bs_256_epochs_50.h5"
MODEL_PATH = 'model/82.66/'

import sys
sys.path.append(MODEL_PATH)
from pre_model import model


# 参数以及模型对应的超参数
TEST_NAME = glob.glob("data/Test_A/" + "*.jpg")
OPT = 'adam'
LOSS = 'binary_crossentropy'
dropout_ALPHA = 0.4
BATCH_SIZE = 256
EPOCHS = 50
input_shape = (150, 150, 3)
IMAGE_SIZE = (150, 150)


# 加载模型和参数
def load_model():
	pre_model = model(input_shape, OPT, LOSS, dropout_ALPHA)
	pre_model.load_weights(MODEL_PATH + MODEL_NAME)

	return pre_model


# 加载测试集及处理
def load_and_process_test():
	test = []

	for filename in TEST_NAME:
		test_img = image.load_img(filename, target_size=IMAGE_SIZE)
		test_img = image.img_to_array(test_img)
		test.append(test_img)

	test = np.array(test, dtype=np.float32)
	test = test / 255

	return test


# 预测
def predict(test, pre_model):
	result = pre_model.predict(test.reshape(len(test),\
							   IMAGE_SIZE[0], \
							   IMAGE_SIZE[1], \
							   3))
	result = list(result.reshape(len(result),))

	return result


# 处理测试集图片名称
def test_name_process():
	global TEST_NAME

	for n, i in enumerate(TEST_NAME):
		TEST_NAME[n] = "pic_" + i.lstrip("data/Test_A/").rstrip(".jpg")


# 保存为csv
def save_csv(result):
	result_table = pd.DataFrame({"pic_id": TEST_NAME, "pred": result})
	result_table.to_csv("predict.csv", index=False)


if __name__ == "__main__":
	pre_model = load_model()
	test = load_and_process_test()
	result = predict(test, pre_model)
	test_name_process()
	save_csv(result)
