import csv
import numpy as np
import deepcut
from gensim.models import Word2Vec
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Dropout, Concatenate
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix

#------------------Import data set---------------------------

file = open('input_T.txt', 'r',encoding = 'utf-8-sig')
data1 = list(csv.reader(file, delimiter=':'))
file = open('ans_T.txt', 'r',encoding = 'utf-8-sig')
data2 = list(csv.reader(file, delimiter=':'))
data = []
for i in range(len(data1) - 280):
    if(data2[i][2] == 'H'):
        j = 0
    elif(data2[i][2] == 'P'):
        j = 1
    else:
        j = 2
    data.append([j, data1[i][2]])
shuffle(data)
for i in range(280):
    if(data2[i+2083][2] == 'H'):
        j = 0
    elif(data2[i+2083][2] == 'P'):
        j = 1
    else:
        j = 2
    data.append([j, data1[i+2083][2]])
file = open('input.txt', 'r', encoding = 'utf-8-sig')
data1 = list(csv.reader(file, delimiter=':'))
for i in range(len(data1)):
    data.append([0, data1[i][2]])

labels = [int(d[0]) for d in data]
sentences = [d[1] for d in data]
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
max_sentence_length = max([len(s) for s in words])
print(max_sentence_length)

#------------------Extract word vectors---------------------------

word_vector_length = 16
wvmodel = Word2Vec(words, size=word_vector_length, window=5, min_count=1, sg=1)

word_vectors = np.zeros((len(words),max_sentence_length,word_vector_length))
sample_idx = 0
for s in words: #for each sentence
    word_idx = max_sentence_length - len(s) #ประโยที่สั่นกว่าความยาวสูงสุด word vector ของต้นประโยคจะเป็น 0
    for w in s: #for each word in a sentence
        word_vectors[sample_idx,word_idx,:] = wvmodel.wv[w]
        word_idx = word_idx+1
    sample_idx = sample_idx+1

print(word_vectors.shape)
print(word_vectors[:2])
    
#------------------Create and train RNN---------------------------    

inputLayer = Input(shape=(max_sentence_length,word_vector_length,))
#srnn = SimpleRNN(30, activation='relu')(inputLayer) #the number of nodes in hidden layer = 32
hiddenLayer1 = Dense(30, activation='relu')(inputLayer)
hiddenLayer2 = Dense(20, activation='relu')(hiddenLayer1)
hiddenLayer3 = Dense(10, activation='relu')(hiddenLayer1)
hiddenLayer4 = Concatenate()([hiddenLayer1, hiddenLayer2])
hiddenLayer5 = Concatenate()([hiddenLayer1, hiddenLayer3])
hiddenLayer6 = Dense(30, activation='relu')(hiddenLayer4)
hiddenLayer7 = Dense(20, activation='relu')(hiddenLayer5)
hiddenLayer8 = Concatenate()([hiddenLayer6, hiddenLayer7])
hiddenLayer9 = Dense(20, activation='relu')(hiddenLayer8)
hiddenLayer10 = Dense(20, activation='relu')(hiddenLayer5)
hiddenLayer11 = Concatenate()([hiddenLayer9, hiddenLayer10])
rnn = GRU(20, activation='relu')(hiddenLayer11)
#rnn = LSTM(30, activation='relu')(hiddenLayer8)
outputLayer = Dense(3, activation='softmax')(rnn) #for 3 classes
model = Model(inputs=inputLayer, outputs=outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(word_vectors[:2082], to_categorical(labels[:2082]), epochs=150, batch_size=70, validation_split = 0.1)

#------------------Evaluate by test set---------------------------  

y_pred = model.predict(word_vectors[2083:2363])

cm = confusion_matrix(labels[2083:2363], y_pred.argmax(axis=1))
print('Confusion Matrix')
print(cm)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

model.save('project_RNN.h5')

#print(y_pred.argmax(axis=1))

y_pred = model.predict(word_vectors[2363:])

file = open('ans.txt', 'w', encoding = 'utf-8-sig')
j = 1
for i in y_pred.argmax(axis=1):
    if(i == 0):
        ans = 'H'
    elif(i == 1):
        ans = 'P'
    else:
        ans = 'M'
    file.write(str(j) + '::' + ans + '\n')
    j+=1
file.close()