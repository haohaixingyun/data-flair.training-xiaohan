import nltk
# Ethan -> NLTK 模块是一个巨大的工具包，目的是在整个自然语言处理（NLP）方法上帮助您。 NLTK 将为您提供一切，从将段落拆分为句子，
# 拆分词语，识别这些词语的词性，高亮主题，甚至帮助您的机器了解文本关于什么。
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Ethan ->  NLTK 词形还原   print(lemmatizer.lemmatize("cats"))  --->  cat
import json
import pickle

'''
pickle模块详解

该pickle模块实现了用于序列化和反序列化Python对象结构的二进制协议。 “Pickling”是将Python对象层次结构转换为字节流的过程， “unpickling”是反向操作，从而将字节流（来自二进制文件或类似字节的对象）转换回对象层次结构。pickle模块对于错误或恶意构造的数据是不安全的。

pickle协议和JSON（JavaScript Object Notation）的区别 ：

　　1. JSON是一种文本序列化格式（它输出unicode文本，虽然大部分时间它被编码utf-8），而pickle是二进制序列化格式;

　　2. JSON是人类可读的，而pickle则不是;

　　3. JSON是可互操作的，并且在Python生态系统之外广泛使用，而pickle是特定于Python的;

默认情况下，JSON只能表示Python内置类型的子集，而不能表示自定义类; pickle可以表示极其庞大的Python类型（其中许多是自动的，通过巧妙地使用Python的内省工具;复杂的案例可以通过实现特定的对象API来解决）。

pickle 数据格式是特定于Python的。它的优点是没有外部标准强加的限制，例如JSON或XDR（不能代表指针共享）; 但是这意味着非Python程序可能无法重建pickled Python对象。

默认情况下，pickle数据格式使用相对紧凑的二进制表示。如果您需要最佳尺寸特征，则可以有效地压缩数据。
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)    # Ethan 按照词来分词
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model_my.h5', hist)

print("model created")
