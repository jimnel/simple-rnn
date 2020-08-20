# simple-rnn
A simple implementation of a recurrent neural network (RNN)

It can be difficult to find a good data-set to train a computationally inexpensie RNN on. 
Here I create a classification data-set by using sequences of numbers between 0 and 1 and classifying them as
0 if they are asscending (A), 1 if they are decsending (D) and 2 if neither (2).

In the modules folder there is code to create the data and network. The net is implemented in PyTorch.

'eda.py' plots the distribution of the data shown in 'eda.png'.

'train.py' trains the network with the history of the network plotted in 'history.py' (Figure 'history.png')

Finally we test the model on much larger sequences then it was trained on ('generalizaton.py'), finding that it can extrapolate to a sequence size of 10 
(the maximum it was trained on was 6) before becoming inaccurate as shown in Figure 'generalization.png'.
