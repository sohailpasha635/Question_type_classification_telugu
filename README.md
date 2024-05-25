                               # Question Type Classification for Telugu


List of contents:

1) Description

2) How to use the project

3) Libraries Required

4) Feature Extraction Techniques

5) Implementation and Results
  
  5.1 : Statistical Models

  5.2 : Deep Laerning Neural Networks
  
  5.3 : BERT

6)Credits/contributors



1)Description:

-> The objective of this project is to identify and classify unlabeled telugu question/query to its corresponding NER type.
i.e., to predict the answer type to the given input query/question.

-> We have implemented this project by training statistical models (Naive Bayes, Multi Layer Perceptron,Support Vector Machine,
 Linear Regression, Random Forest) and Deep Learning Neural Networks(CNN, Bidirectional_RNN, RNN_LSTM, RNN_GRU) by providing 
 Corpus(which contains labelled Telugu Queries) as input to them.

-> Corpus contains telugu queries along with their corresponding NER type labels. Corpus contains eight different labels such
 as Person, Money, Organisation, Location, Number, Date, Time, Percentage.

-> The dataset which we used is in the "data" folder.

-> We have also implented using BERT(Bidiectional Encoder Representations from Transformers)

-> Implementation using Statistical Models is done in - Question_Type_Classification_Statistical.ipynb file

-> Implementation using Deep Learning Neural Networks is done in - Question_Type_Classification_DeepLearning.ipynb file

-> Implementation using BERT is done in - Question_Type_Classification_BERT.ipynb file



2)How to use the project:

-> Users can implement the code using jupyter notebook(except BERT) and google colaboratory(colab).





3)Libraries used:

-> we need to install(in case of jupyter notebook) and import some libraries in order to implement the code.

-> They are:
   -> Pandas
   -> Scikit-learn
   -> Keras
   -> Numpy
   -> Transformers
   -> Tensorflow

-> Upgrade pip version to 3.8.6(ignore if already upgraded)




4)Feature Extraction Techinques:

-> In order to give the data as input to the statistical models or Deep learning neural networks, data has to be tranformed
   into falttened features so that we can give that flattened features as input to the models.

-> We transformed the data into TFIDF(Term frequency-inverse document frequency) vectors using 3 different types(word level, 
   char level, n-gram level) which can be used as input features for statistical models.

-> we tranformed the data into embedding matrix using pretrained FastText and BytePair embeddings which can be used as input
   features for Deep learning neural networks.



5)Implementation and Results:

-> We have implemented the project using statistical approach, Deep Learning Neural network approach and BERT(Bidirectioanal 
  Encoder Representations from tranformers) approach.

5.1)Statistical approach:

-> Statistical modelling is the process of applying statictical analysis to a dataset in order to prepare data ready for 
   modelling.
-> A statistical model is the mathematical representation of raw data.

-> We have different models in this approach such as Logistic regression, Multilayer perceptron, Naive Bayes, Random Forest, 
   Support vector machine

i) Logistic regression:-

-> Logistic regression is a Supervised Learning technique which is used for predicting the categorical dependent variable 
  using a given set of independent variables.

-> Accracies using Logistic regression classifier :-
   
   WordLevel tfidf      -  76
   N-gram level tfidf   -  78
   Char-level tfidf     -  82
   Count level vectors  -  73


ii) Support Vector Machine:-

-> Support vector machines are supervised learning models with associated learning algorithms that analyze data for 
  classification and regression analysis

-> Accuracies using Support Vector Machine classifier :-

   WordLevel tfidf      -  81
   N-gram level tfidf   -  84
   Char-level tfidf     -  88
   Count level vectors  -  72


iii) MultiLayer Perceptron:-

-> Multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN) which consists of at least 
  three layers of nodes: an input layer, a hidden layer and an output layer and utilizes a supervised learning technique 
  called backpropagation for training.

-> Accuracies using Multi Layer Perceptron classifier :-

   WordLevel tfidf      -  76
   N-gram level tfidf   -  81
   Char-level tfidf     -  85
   Count level vectors  -  72


iv) Naive Bayes:-

-> It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.

-> Accuracies using Naive bayes classsifier :-

   WordLevel tfidf      -  74
   N-gram level tfidf   -  74
   Char-level tfidf     -  78
   Count level vectors  -  74


v) Random Forest:-

-> Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and 
   takes the average to improve the predictive accuracy of that dataset.
 
->Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority 
  votes of predictions, and it predicts the final output.

-> Accuracies using Random Forest classifier :-

   WordLevel tfidf      -  79
   N-gram level tfidf   -  80
   Char-level tfidf     -  83
   Count level vectors  -  77


5.2) Neural Network Approach:

-> Deep Neural Networks are more complex neural networks in which the hidden layers perform much more complex operations 
   than simple sigmoid or relu activations.

-> We used Pretrained FastText and Byte Pair Word embeddings as features for neural networks.

-> Users can find the pretrained telugu FastText and Byte Pair Word embeddings in the following websites:
  
   https://fasttext.cc/docs/en/crawl-vectors.html  - FastText Word embeddings
   https://bpemb.h-its.org/te/                     - Byte Pair Word embeddings
   

-> We have implemented our project using various neural networks such as Convolutional Neural Network (CNN), 
  Long short-term memory recurrent neural network, Gated Recurrent Unit recurrent neural network, Bidirectional recurrent
  neural network


i) Convolutional Neural Network (CNN):-

-> A standard model for document classification is to use an Embedding layer as input, followed by a one-dimensional 
  convolutional neural network

-> Convolutional networks are a specialized type of neural networks that use convolution in place of general 
  matrix multiplication in at least one of their layers .

-> Accuracies using CNN :-

   FastText word embeddings      -  90
   Byte pair word embeddings     -  85


ii) Long short-term memory recurrent neural network (LSTM RNN):-

->Long Short Term Memory is a kind of recurrent neural network which takles the problem of long-term dependencies of RNN 
  in which the RNN cannot predict the word stored in the long term memory but LSTM can by default retain the information 
  for long period of time and predict more accurately than RNN.

-> Accuracies using LSTM :-

   FastText word embeddings      -  88
   Byte pair word embeddings     -  86


iii) Gated Recurrent Unit recurrent neural network(GRU):-

-> GRU is a variant of the RNN architecture, and uses gating mechanisms to control and manage the flow of information 
  between cells in the neural network.

-> The structure of the GRU allows it to adaptively capture dependencies from large sequences of data without discarding 
  information from earlier parts of the sequence using gating units.

-> Accuracies using GRU :-

   FastText word embeddings      -  89
   Byte pair word embeddings     -  85


iv) Bidirectional recurrent neural network(Bi_RNN):-

-> Bidirectional recurrent neural networks (BRNN) connect two hidden layers of opposite directions to the same output.

-> BY this the output layer can get information from past (backwards) and future (forward) states simultaneously.

-> Accuracies obtained using Bi_RNN :-

   FastText word embeddings      -  89
   Byte pair word embeddings     -  85



5.3) BERT(Bidirectioanal Encoder Representations from tranformers) approach:

-> BERT is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by 
Google.

-> We implemented it in colab by mounting a drive and through drive we accessed our dataset.

-> Accuracy obtained with BERT  - 88




6)Credits/Contributors:

-> This project is implemented by:
   
   1) Sohail Pasha Md          - IIIT BASAR
   2) Akanksha Anandham        - IIIT BASAR 
   3) Sai Srujana Santhapuri   - IIIT BASAR
   4) Suchita Anumula          - IIIT BASAR

-> Under the Mentorship of URLANA ASHOK (LRTC IIIT Hyderabad)


