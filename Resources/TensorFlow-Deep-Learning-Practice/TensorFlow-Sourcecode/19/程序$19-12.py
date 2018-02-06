import tensorflow as tf

def batch_read_and_decode(filename,img_heigh=224,img_width=224,batchSize=100):
    # 创建文件队列
    fileNameQue = tf.train.string_input_producer([filename], shuffle=True)
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(fileNameQue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [img_heigh, img_width, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    min_after_dequeue = batchSize * 9
    capacity = min_after_dequeue + batchSize
    # 预取图像和label并随机打乱，组成batch，此时tensor rank发生了变化，多了一个batch大小的维度
    exampleBatch,labelBatch = tf.train.shuffle_batch([img, label],batch_size=batchSize, capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue)
    return exampleBatch,labelBatch


if __name__ == "__main__":
   init = tf.initialize_all_variables()
    exampleBatch, labelBatch = batch_read_and_decode("train.tfrecords")

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            example, label = sess.run([exampleBatch, labelBatch])
            print(example[0][112],label)
            print("---------%i---------"%i)

        coord.request_stop()
        coord.join(threads)
