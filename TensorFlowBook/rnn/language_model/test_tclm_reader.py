from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tclm_reader
import tensorflow as tf


class TCReaderTest(tf.test.TestCase):
    def setUp(self):
        self._string_data = "\n".join(
            [" hello there i am",
             " rain as day",
             " want some cheesy puffs ?"])

    def testTCProducer(self):
        raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
        batch_size = 3
        num_steps = 2
        x, y = tclm_reader.tensorflow_code_producer(raw_data, batch_size,
                                                    num_steps)
        with self.test_session() as session:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(session, coord=coord)
            try:
                xval, yval = session.run([x, y])
                print(xval)
                print(yval)
                self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
                self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
                xval, yval = session.run([x, y])
                print(xval)
                print(yval)
                self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
                self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
            finally:
                coord.request_stop()
                coord.join()

    def testTCProducer(self):
        raw_data = [257, 1, 2, 3, 4, 9, 10, 11, 15, 16, 17, 5, 6, 12, 13, 14, 7,
                    8, 18, 19, 20, 258]
        batch_size = 3
        num_steps = 4
        x, y = tclm_reader.tensorflow_code_producer(raw_data, batch_size,
                                                    num_steps)
        with self.test_session() as session:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(session, coord=coord)
            try:
                xval, yval = session.run([x, y])
                print(xval)
                print(yval)

                xval, yval = session.run([x, y])
                print(xval)
                print(yval)
            finally:
                coord.request_stop()
                coord.join()


if __name__ == "__main__":
    tf.test.main()
