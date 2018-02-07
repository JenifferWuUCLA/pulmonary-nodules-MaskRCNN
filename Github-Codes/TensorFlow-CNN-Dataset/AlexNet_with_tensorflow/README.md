# AlexNet-with-tensorflow
an easy implement of AlexNet with tensorflow, which has a detailed explanation.

<img src="https://raw.githubusercontent.com/hjptriplebee/AlexNet_with_tensorflow/master/testModel/005525.jpg" width = "200" height = "150" alt="alexnet" /><img src="https://raw.githubusercontent.com/hjptriplebee/AlexNet_with_tensorflow/master/testModel/002689.jpg" width = "200" height = "150" alt="alexnet" /><img src="https://raw.githubusercontent.com/hjptriplebee/AlexNet_with_tensorflow/master/testModel/000018.jpg" width = "200" height = "150" alt="alexnet" />

<img src="https://raw.githubusercontent.com/hjptriplebee/AlexNet_with_tensorflow/master/demo1.png" width = "200" height = "150" alt="tensorflow" /><img src="https://raw.githubusercontent.com/hjptriplebee/AlexNet_with_tensorflow/master/demo2.png" width = "200" height = "150" alt="tensorflow" /><img src="https://raw.githubusercontent.com/hjptriplebee/AlexNet_with_tensorflow/master/demo3.png" width = "200" height = "150" alt="tensorflow" />

The code is an implement of AlexNet with tensorflow. The detailed explanation can be found [here](http://blog.csdn.net/accepthjp/article/details/69999309)

Before running the code, you should confirm that you have :

- Python (2 and 3 is all ok, 2 need a little change on function"print()")
- tensorflow 1.0
- opencv

Then, you should download the model file "bvlc_alexnet.npy" which can be found [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)or [here](http://pan.baidu.com/s/1c1ULewC)(for users who can't download from the first link).

Finally, run the test file with "**python3 testModel.py folder testModel**", you will see some images with the predicted label (press any key to move on to the next image).

The command also **supports url**. 

For eg. "**python3 testModel.py url http://www.cats.org.uk/uploads/images/featurebox_sidebar_kids/Cat-Behaviour.jpg**"

You can also use tensorboard to monitor the process. Remeber to see [detailed explanation](http://blog.csdn.net/accepthjp/article/details/69999309).

<br />
<br />

If you have any problem, please contact me!

blog  ：[http://blog.csdn.net/accepthjp](http://blog.csdn.net/accepthjp)

email ：huangjipengnju@gmail.com
