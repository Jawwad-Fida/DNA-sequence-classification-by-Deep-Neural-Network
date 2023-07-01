# Overview

In this work, we have worked on the DNA sequence classification problem where the input is the DNA sequence and the output class states whether a certain histone protein is present on the sequence or not. For this purpose, we have used one of the datasets from 12 different datasets that we have collected. The name of the dataset is H3K4me2. Because of the time limitation, just one dataset was chosen, and H3K4me2 was chosen because prior research on this dataset had shown unsatisfactory results

To represent a sequence, we have utilized k-mer representation. For the sequence embedding we have used one-hot encoding, and two different word embedding models, one is Word2Vec and another one is BERT.

**Domain**: Bioinformatics, Deep Learning, Data Science.

## Contributors
- **Team Members:** [Md. Tarek Hasan](https://www.linkedin.com/in/tarekhasan/), [Mohammed Jawwadul Islam](https://www.linkedin.com/in/jawwadfida/), [Md Fahad Al Rafi](https://www.linkedin.com/in/md-fahad-al-al-rafi-14b968111/), [Arifa Akter](https://www.linkedin.com/in/arifa-a-asha/), [Sumayra Islam](https://www.linkedin.com/in/sumayra-islam-orni-b91aaa219/),
- **Course:** Bioinformatics Project - B.Sc. in Computer Science and Engineering (CSE)
- **Project Duration:** Fall 2021 Trimester (Nov 2021 - Jan 2022)
- **Dataset from:** Nguyen who is one the authors of the paper titled “DNA sequence classification by convolutional neural network” 


## Installing software and files
To do the project, we need to install some softwares and files. In this regard, we will be doing all the implementations in Python language on jupyter notebook. To install jupyter notebook and launch other application and files at first we have to download Anaconda which is free.

Link to Download Anaconda : https://www.anaconda.com/?modal=nucleus-commercial

Guideline for installing Anaconda : https://www.geeksforgeeks.org/how-to-install-anaconda-on-windows/

Once Anaconda is downloaded and installed successfully, we may proceed to download Jupyter notebook.

## Download and Install Jupyter Notebook
Link to download Jupyter using Anaconda : https://docs.anaconda.com/ae-notebooks/4.3.1/user-guide/basic-tasks/apps/jupyter/

More informations : https://mas-dse.github.io/startup/anaconda-windows-install/

Guideline to use Jupyter notebook : https://www.dataquest.io/blog/jupyter-notebook-tutorial/

## Using Google Colaboratory
For implementing the project with no configuration we can use Google Colaboratory as well.

## Installing Python libraries and packages
The required python libraries and packages are,
- pandas
- numpy
- keras
- sklearn
- tensorflow

# Features of the Dataset
DNA sequences wrapped around histone proteins are the subject of datasets

* For our experiment, we selected one of the datasets entitled H3K4me2. 
* H3K4me2 has 30683 DNA sequences whose 18143 samples fall under the positive class, the rest of the samples fall under the negative class, and it makes the problem binary class classification. 
* The ratio of the positive-negative class is around (59:41)%. 
* The class label represents the presence of H3K4me2 histone proteins in the sequences. 
* The base length of the sequences is 500.


# Data Preprocessing
* The datasets were gathered in.txt format. We discovered that the dataset contains id, sequence, and class label during the Exploratory Data Analysis phase of our work. 
* We dropped the id column from the dataset because it is the only trait that all of the samples share. 
* Except for two samples, H3K4me2 includes 36799 DNA sequences, the majority of which are 500 bases long. Those two sequences have lengths of 310 and 290, respectively. To begin, we employed the zero-padding strategy to tackle the problem. However, because there are only two examples of varying lengths, we dropped those two samples from the dataset later for experiments, as these samples may cause noise.
* we have used the K-mer sequence representation technique to represent a DNA sequence, we have used the K-mer sequence representation technique
* For sequence emdedding after applying the 3-mer representation technique, we have experimented using different embedding techniques. The first three embedding methods are named SequenceEmbedding1D, SequenceEmbedding2D, SequenceEmbedding2D_V2, Word2Vec and BERT.
  * SequenceEmbedding1D is the one-dimensional representation of a single DNA sequence which is basically the one-hot encoding. 
  * SequenceEmbedding2D is the two-dimensional representation of a single DNA sequence where the first row is the one-hot encoding of a sequence after applying 3-mer representation. The second row is the one-hot encoding of a left-rotated sequence after applying 3-mer representation. 
  * the third row of SequenceEmbedding2D_V2 is the one-hot encoding of a right-rotated sequence after applying 3-mer representation.
  * Word2Vec and BERT are the word embedding techniques for language modeling.
  
## SequenceEmbedding1D

```python
import numpy as np
class SeqEmbedding1D():
    def __init__(self):
        self.Dictionary = {}

    def Sequence_to_Numeric(self,k,sequence):
        if k==0:
            temp = [0] * self.size_of_vector
            temp[len(self.Dictionary)] = 1
            self.Dictionary[sequence] = temp
            return
        nucleotide = ['A','C','G','T']
        for n in nucleotide:
            self.Sequence_to_Numeric(k-1,sequence+n)
        return

    def fit(self, sequences, window_size, stride_size):
        self.size_of_vector = 4 ** window_size
        self.Sequence_to_Numeric(window_size,"")

        vectorized = []

        for seq in sequences:
            first_layer_embedding = []
            for k in range(window_size, len(seq)+1, stride_size):
                try:
                    first_layer_embedding.append(self.Dictionary[seq[k-window_size:k]])
                except:
                    first_layer_embedding.append([0]*self.size_of_vector)
            
            vector0 = []
            for i in range(len(first_layer_embedding)):
                vector0+=first_layer_embedding[i]

            vectorized.append(vector0)
        
        # Handling inequal length problem using zero padding
        max_len = 0
        for vec in vectorized:
            max_len = max(max_len, len(vec))
        for i in range(len(vectorized)):
            required = max_len - len(vectorized[i])

            vectorized[i]+=([0]*required)
            vectorized[i] = np.array(vectorized[i])
            
        return np.array(vectorized)
```

## Sequence Embedding 2D

```python
import numpy as np
class SeqEmbedding2D(SeqEmbedding1D):
    def __init__(self):
        super().__init__()

    def fit(self, sequences, window_size, stride_size):
        self.size_of_vector = 4 ** window_size
        super().Sequence_to_Numeric(window_size,"")

        vectorized = []

        for seq in sequences:
            first_layer_embedding = []
            for k in range(window_size, len(seq)+1, stride_size):
                try:
                    first_layer_embedding.append(self.Dictionary[seq[k-window_size:k]])
                except:
                    first_layer_embedding.append([0]*self.size_of_vector)
            
            vector0 = []
            vector1 = []
            for i in range(len(first_layer_embedding)):
                if i>0:
                    vector1+=first_layer_embedding[i]
                vector0+=first_layer_embedding[i]
            vector1+=first_layer_embedding[0]

            vectorized.append([vector0, vector1])
        
        # Handling inequal length problem using zero padding
        max_len = 0
        for vec in vectorized:
            max_len = max(max_len, len(vec[0]))
        for i in range(len(vectorized)):
            required = max_len - len(vectorized[i][0])

            vectorized[i][0]+=([0]*required)
            vectorized[i][0] = np.array(vectorized[i][0])

            vectorized[i][1]+=([0]*required)
            vectorized[i][1] = np.array(vectorized[i][1])

            vectorized[i] = np.array(vectorized[i])
            
        return np.array(vectorized)

```

## Applying BERT for Sequence Embedding

```python
!pip install git+https://github.com/khalidsaifullaah/BERTify

from bertify import BERTify
class BERT_Seq(Word2Vec_Seq):
    def __init__(self):
        super().__init__()
        self.en_bertify = BERTify(
            lang="en",
            last_four_layers_embedding=True
        )
    def fit(self,sequences):
        sequence_to_sentences = super().SeqtoSen(sequences)
        en_embeddings = self.en_bertify.embedding(sequence_to_sentences)
        return en_embeddings

instance = BERT_Seq()
arrays = instance.fit(sequences[:500])
arrays.shape
```

# Deep Learning Models

After the completion of sequence embedding, we have used deep learning models for the classification task. We have used two different deep learning models for this purpose, one is  and the other is . 

## 1D Convolutional Neural Network (CNN)

```python
def get_model(X):
    model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv1D(filters=16, kernel_size=4,activation='relu',input_shape=(X.shape[1], 1)), 
                                    tf.keras.layers.MaxPooling1D(pool_size=2),
                                    tf.keras.layers.Conv1D(filters=16,kernel_size=4,activation='relu'), 
                                    tf.keras.layers.MaxPooling1D(pool_size=2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100,activation='relu'), 
                                    tf.keras.layers.Dropout(0.5), 
                                    tf.keras.layers.Dense(2,activation='softmax') 
    ])
    print(model.summary())
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    from keras.utils.vis_utils import plot_model
    plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)
    return model

def training(train_X, train_y, val_X, val_y, filename_best_model, first_time):
    train_X = train_X.reshape(train_X.shape[0], -1)
    val_X = val_X.reshape(val_X.shape[0], -1)
    
    if first_time == False:
        model = tf.keras.models.load_model(filename_best_model)
    else:
        model = get_model(train_X)
        
    es = EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=200)
    mc = ModelCheckpoint(filename_best_model,monitor='val_accuracy',mode='max',verbose=1,save_best_only=True, save_weights_only=False)

    model.fit(train_X,train_y, validation_data=(val_X,val_y), epochs=300,batch_size=256,callbacks=[es,mc])
    return
```

## Bidirectional Long Short-Term Memory (BiLSTM)

```python
def get_model():
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = 256, input_length=498))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(units=256,return_sequences = True)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model
    
es = EarlyStopping(monitor='val_accuracy',mode='max',verbose=1,patience=200)
mc = ModelCheckpoint('best_model.h5',monitor='val_accuracy',mode='max',verbose=1,save_best_only=True, save_weights_only=True)  
```

# Experimental Analysis

## Predictions

After the data cleaning phase, we had 36797 samples. We have used 80% of the whole dataset for training and the rest of the samples for testing. The dataset has been split using train_test_split from sklearn.model_selection stratifying by the class label. We have utilized 10% of the training data for validation purposes. For the first five experiments we have used batch training as it was throwing an exception of resource exhaustion.

```python
def testing(test_X, test_y, filename_best_model, output_file):
    test_X = test_X.reshape(test_X.shape[0], -1)

    # model = get_model(test_X)
    model = tf.keras.models.load_model(filename_best_model)

    p = model.predict(test_X)
    prediction = []
    for x in p:
        if(x[0]>x[1]):
            prediction.append(0)
        else:
            prediction.append(1)

    f = open(output_file, "w")
    f.write(f'Accuracy: {accuracy_score(test_y,prediction)}\n')
    f.write(f'Precision: {precision_score(test_y,prediction)}\n')
    f.write(f'Recall: {recall_score(test_y,prediction)}\n')
    f.write(f'F1 Score: {f1_score(test_y,prediction)}\n')
    f.write(f'MCC Score: {matthews_corrcoef(test_y,prediction)}\n')
    f.close()
    return
```

## Results

The evaluation metrics we used for our experiments are accuracy, precision, recall, f1-score, and Matthews Correlation Coefficient (MCC) score. The minimum value of accuracy, precision, recall, f1-score can be 0 and the maximum value can be 1. The minimum value of the MCC score can be -1 and the maximum value can be 1.

<img src="https://user-images.githubusercontent.com/64092765/153142213-0bf682a5-ce05-4977-8d71-97fc266f6abe.png" width="75%">

MCC score 0 indicates the model's randomized predictions. The recall score indicates how well the classifier can find all positive samples. We can say that the model's ability to classify all positive samples has been at an all-time high over the last five experiments. The highest MCC score we received was 0.1573, indicating that the model is very near to predicting in a randomized approach. We attain a maximum accuracy of 60.27%, which is much lower than the state-of-the-art result of 71.77%. To improve the score, we need to emphasize more on the sequence embedding approach. Furthermore, we can experiment with various deep learning techniques.

## Project Report

[Bioinformatics Project Report](https://github.com/Jawwad-Fida/DNA-sequence-classification-by-Deep-Neural-Network/files/11925655/Bioinformatics.Project.Report.of.Group.1.docx.pdf)


