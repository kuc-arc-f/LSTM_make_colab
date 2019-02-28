# encoding: utf-8

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model
from janome.tokenizer import Tokenizer
import numpy as np
import random
import sys
import io

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#
def get_token(text):
    text =Tokenizer().tokenize(text, wakati=True)  # 分かち書きする
    return text

#data

#s= get_token("本日は、朝早く起きてました。")
#print(s)
#quit()
maxlen = 5
step = 1
sentences = []
next_chars = []
path = './data.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

text =Tokenizer().tokenize(text, wakati=True)  # 分かち書きする
#print(text[0 :5])
#quit()

chars = text
count = 0
char_indices = {}  # 辞書初期化
indices_char = {}  # 逆引き辞書初期化

for word in chars:
    if not word in char_indices:  # 未登録なら
       char_indices[word] = count  # 登録する      
       count +=1
#       print(count,word)  # 登録した単語を表示
# 逆引き辞書を辞書から作成する
indices_char = dict([(value, key) for (key, value) in char_indices.items()])

start_index = 0
#quit()
#
print('Build model...')
model=load_model('model.h5')

#pred
for diversity in [0.2]:  # diversity は 0.2のみ使用 
    print('----- diversity:', diversity)
    generated = ''
    text= get_token("どんなつまらない仕事でも楽しんでやるのだ")
    sentence = text[start_index: start_index + maxlen]
    generated += "".join(sentence)
    print(sentence )
    print('----- Generating with seed: "' + "".join(sentence)+ '"')
    sys.stdout.write(generated)

    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:]
        # sentence はリストなので append で結合する
        sentence.append(next_char)  

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
