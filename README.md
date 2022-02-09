# DNA sequence classification by Deep Neural Network: Project Overview
* worked on the DNA sequence classification problem where the input is the DNA sequence and the output class states whether a certain histone protein is present on the sequence or not.
* used one of the datasets from 12 different datasets that we have collected. The name of the dataset is H3K4me2 
* To represent a sequence, we have utilized k-mer representation
* For the sequence embedding we have used one-hot encoding
* Different word embedding models: Word2Vec, BERT, Keras Embedding layer, Bi-LSTM, and CNN 

## Bioinformatics Project - B.Sc. in Computer Science and Engineering (CSE)

## Created by: - Md. Tarek Hasan, Mohammed Jawwadul Islam, Md Fahad Al Rafi, Arifa Akter, Sumayra Islam

### Date of Completion: - Fall 2021 Trimester (Nov 2021 - Jan 2022)

### [Linkedin of Jawwadul](https://www.linkedin.com/in/jawwadfida/)  
### [Linkedin of Tarek](https://www.linkedin.com/in/tarekhasan/)
### [Linkedin of Fahad](https://www.linkedin.com/in/md-fahad-al-al-rafi-14b968111/)
### [Linkedin of Arifa](https://www.linkedin.com/in/arifa-a-asha/)
### [Linkedin of Sumayra](https://www.linkedin.com/in/sumayra-islam-orni-b91aaa219/)


## Code and Resources Used 
**Python Version:** 3.7.11  
**Packages:** numpy, pandas, keras, tensorflow, sklearn
**Dataset from:** Nguyen who is one the authors of the paper titled “DNA sequence classification by convolutional neural network”  <br>  

## Features of the Dataset
DNA sequences wrapped around histone proteins are the subject of datasets

* For our experiment, we selected one of the datasets entitled H3K4me2. 
* H3K4me2 has 30683 DNA sequences whose 18143 samples fall under the positive class, the rest of the samples fall under the negative class, and it makes the problem binary class classification. 
* The ratio of the positive-negative class is around (59:41)%. 
* The class label represents the presence of H3K4me2 histone proteins in the sequences. 
* The base length of the sequences is 500.


## Data Preprocessing
* The datasets were gathered in.txt format. We discovered that the dataset contains id, sequence, and class label during the Exploratory Data Analysis phase of our work. 
* We dropped the id column from the dataset because it is the only trait that all of the samples share. 
* Except for two samples, H3K4me2 includes 36799 DNA sequences, the majority of which are 500 bases long. Those two sequences have lengths of 310 and 290, respectively. To begin, we employed the zero-padding strategy to tackle the problem. However, because there are only two examples of varying lengths, we dropped those two samples from the dataset later for experiments, as these samples may cause noise.
* we have used the K-mer sequence representation technique to represent a DNA sequence, we have used the K-mer sequence representation technique
* For sequence emdedding after applying the 3-mer representation technique, we have experimented using different embedding techniques. The first three embedding methods are named SequenceEmbedding1D, SequenceEmbedding2D, SequenceEmbedding2D_V2, Word2Vec and BERT.
  * SequenceEmbedding1D is the one-dimensional representation of a single DNA sequence which is basically the one-hot encoding. 
  * SequenceEmbedding2D is the two-dimensional representation of a single DNA sequence where the first row is the one-hot encoding of a sequence after applying 3-mer representation. The second row is the one-hot encoding of a left-rotated sequence after applying 3-mer representation. 
  * the third row of SequenceEmbedding2D_V2 is the one-hot encoding of a right-rotated sequence after applying 3-mer representation.
  * Word2Vec and BERT are the word embedding techniques for language modeling.

## Deep Learning Models

After the completion of sequence embedding, we have used deep learning models for the classification task. We have used two different deep learning models for this purpose, one is Convolutional Neural Network (CNN) and the other is Bidirectional Long Short-Term Memory (Bi-LSTM). 

## Experimental Analysis
After the data cleaning phase, we had 36797 samples. We have used 80% of the whole dataset for training and the rest of the samples for testing. The dataset has been split using train_test_split from sklearn.model_selection stratifying by the class label. We have utilized 10% of the training data for validation purposes. For the first five experiments we have used batch training as it was throwing an exception of resource exhaustion.

The evaluation metrics we used for our experiments are accuracy, precision, recall, f1-score, and Matthews Correlation Coefficient (MCC) score. The minimum value of accuracy, precision, recall, f1-score can be 0 and the maximum value can be 1. The minimum value of the MCC score can be -1 and the maximum value can be 1.

![image](https://user-images.githubusercontent.com/64092765/153142213-0bf682a5-ce05-4977-8d71-97fc266f6abe.png)

## Discussion 

MCC score 0 indicates the model's randomized predictions. The recall score indicates how well the classifier can find all positive samples. We can say that the model's ability to classify all positive samples has been at an all-time high over the last five experiments. The highest MCC score we received was 0.1573, indicating that the model is very near to predicting in a randomized approach. We attain a maximum accuracy of 60.27%, which is much lower than the state-of-the-art result of 71.77%. To improve the score, we need to emphasize more on the sequence embedding approach. Furthermore, we can experiment with various deep learning techniques.





