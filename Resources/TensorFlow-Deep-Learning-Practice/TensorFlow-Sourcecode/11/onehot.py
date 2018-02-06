def readFile(filename):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    # 定义Reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    col1, col2, col3, col4, col5 , col6 , col7 = tf.decode_csv(value,record_defaults=record_defaults)
    label = tf.pack([col1,col2])
    features = tf.pack([col3, col4, col5, col6, col7])
    example_batch, label_batch = tf.train.shuffle_batch([features,label],
                                batch_size=3, capacity=100, min_after_dequeue=10)

    return example_batch,label_batch
