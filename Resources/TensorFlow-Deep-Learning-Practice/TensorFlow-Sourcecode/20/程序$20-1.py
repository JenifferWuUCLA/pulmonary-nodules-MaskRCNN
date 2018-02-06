import tensorflow as tf
from datasets import dataset_utils	#这里请读者一定先下载相应的数据集处理文件夹并先导入到工程中

url = "http://download.tensorflow.org/data/flowers.tar.gz"
flowers_data_dir = global_variable.flowers_data_dir
if not tf.gfile.Exists(flowers_data_dir):
    tf.gfile.MakeDirs(flowers_data_dir)

dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)
