#Welcome to the TinyDL - a very small educational neural network simulator

Many deep learning and neural network simulators have been written. Many of them in C++. This one is was created to conduct educational  experiments, which is why it is very small, configurable and extendable. You can create easily feedforward networks, recurrent networks convolutional neural networks for classification and regression.

TinyDL is a fully functional implementation of the backpropagation algorithm with momentum adaption with a variety of transfer functions, simple methods to create networks with connections between input, hidden and output layers, and functions to learn any pattern. If your are interested in a detailed explanation how backpropagation works with an example and the mathematical background, check out [Matt Mazur's post on the topic](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/). If you are interested in Neural Networks and Deep Learning, I recommend reading [Michael Nielsen's online book](http://neuralnetworksanddeeplearning.com/). If you would like to learn more about the two spiral data problem and machine learning solutions that have been researched and published, check out this [paper.](https://pdfs.semanticscholar.org/3d2a/43ce330428822a03df61b3267e19a6c529e2.pdf) And finally, if you just wanted to play around with neural nets and interactively explore the two spiral problem, here is an [interactive online neural net simulator.](http://playground.tensorflow.org/#activation=sigmoid&batchSize=7&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.36963&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification) 

The following three illustrations show how Tiny DL is processing the backpropagation algorithm.

##1. Feed Forward
At first, the total input **net** is the computed by summing up the collective **actual** output from the previous layer. After that, using the activation function F on **net** generates the output **out**. Then **out** is multiplied with the **weight** of the connection to generate the **actual** output for the next layer. The algorithm starts from the input layer, through the hidden layers to the output layer (left to right).
![](https://cloud.githubusercontent.com/assets/17483266/15032402/5105f8a6-1215-11e6-8af1-564cd19c3346.jpg)
##2. Error Backpropagation
This time the algorithm starts from the right side, by summing up the **error** from the previous layer. In the output layer **error_out** is the **target** value minus the **actual** value: error_out = target - out.
The **error_out** multiplied with the **weight** of the connection becomes the error from the previous neuron. After summing all errors up into **error_in**, the derivate of the activation function F' computes **error_out** that is then propagated towards the input layer.   
![](https://cloud.githubusercontent.com/assets/17483266/15032419/7d7cefa2-1215-11e6-9858-0f556de335b7.jpg)
##3. Updating Weights
In the third step all weights are updated by adding a delta value that is computed by multiplying the learning rate **eta** the **error** and the neuron's output **out**. (yes, that's the out value that was computed in step 1)
![](https://cloud.githubusercontent.com/assets/17483266/15032421/8da56882-1215-11e6-815f-dc10e97dfef4.jpg)
##About the Code, Updates and Contribution
There are only three small files to this project. 

1. You can find all classes for this well documented, small neural network simulator in [tinydl.cpp](https://github.com/FrostDataCapital/tinydl/blob/master/tinydl/tinydl.cpp). 

2. The two spiral data set is located in [twospirals.h]  (https://github.com/FrostDataCapital/tinydl/blob/master/tinydl/twospirals.h).

3. And [main.cpp](https://github.com/FrostDataCapital/tinydl/blob/master/tinydl/main.cpp) creates the neural network, learns 
the problem and displays the output function space in ASCII characters.

This project was specifically created for educational purposes, without the speed or bells and whistles of a "production quality" machine learning implementation. If you find bugs, have questions, suggestions or want to contribute, feel free to send me an email.


