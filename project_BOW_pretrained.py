import csv
import numpy as np
import deepcut
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix

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
for sentence in words:
    print(sentence)

#------------------- Extract bag-of-words -------------------------
vocab = set([w for s in words for w in s])
print('Vocab size = '+str(len(vocab)))

bag_of_words = np.zeros((len(words),len(vocab)))
for i in range(0,len(words)):
    count = 0
    for j in range(0,len(words[i])):
        k = 0
        for w in vocab:
            if(words[i][j] == w):
                bag_of_words[i][k] = bag_of_words[i][k]+1
                count = count+1
            k = k+1
    bag_of_words[i] = bag_of_words[i]/count

print(bag_of_words.shape)
print(bag_of_words[0])
    
#------------------Load Model---------------------------    

model = load_model('./project_BOW.h5')
print(model.summary())

#------------------Evaluate by test set---------------------------  

y_pred = model.predict(bag_of_words[2083:2363])

cm = confusion_matrix(labels[2083:2363], y_pred.argmax(axis=1))
print('Confusion Matrix')
print(cm)

#print(y_pred.argmax(axis=1))

y_pred = model.predict(bag_of_words[2363:])

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