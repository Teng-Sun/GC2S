#import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import CKIPClient_python3 as parser
from time import sleep
import os.path
import pickle
from IPython import embed
from opencc import OpenCC
openCC = OpenCC('s2t')

PAD = 3
EOS = 1
UNK = 2
BOS = 0
def sql_to_csv():

    D = pd.read_sql('select * from \"hotel-review\" ', 'sqlite:///HR_30.db')
    del D['id']
    D['review'] = [openCC.convert(sen) for sen in D['review']]
    D.to_csv('data/raw.csv',index=False)

def raw_to_ckip_parse():
    import re
    #if ~os.path.isfile('data/raw.csv'):
    #    raise Exception('No raw csv')
    D = pd.read_csv('data/raw.csv')
    big = []
    f= open('data/seg_data.txt', 'w')
    D = D[pd.notnull(D.review)]
    for string in D['review'] :
        print(string)
        try :
            ans = ' '.join([c[0] for c in parser.parse(re.sub('[,.!]', ' ', string[:200]))])
            big.append(ans)
            print(ans)
        except:
            embed()
            continue
        sleep(0.1)
    #f.close()

    D['review'] = big
    D.to_csv('data/train.csv')


def to_batch(inputs, maxlen=None):
    '''
    Args:
        inputs
            list of sentences
    Outpus :
        shape [maxlen, batch_size] padded with 0s

    '''
    sequence_lengths = [len(s) for s in inputs]
    batch_size = len(inputs)
    if maxlen == None :
        maxlen = max(sequence_lengths)

    batch_major = np.ones(shape=[batch_size, maxlen], dtype=np.int32) * PAD #PAD

    for i, sentence in enumerate(inputs):
        for j, c in enumerate(sentence):
                if j >= maxlen :
                    break
                batch_major[i,j] = c

    time_major = batch_major.swapaxes(0, 1)
    return time_major, sequence_lengths

'''def fit_on_corous(list_):
    try :
        words = list(set(''.join(list_)))
        dict_word = {words[i]:i for i in range(len(words))}
        return dict_word
    except :
        print('please turn nan to \'\'')
        return
def texts_to_sequences(corpus):
'''
import re
def zh(char) :
    if re.match('[\u4e00-\u9fff]+', char) != None :
        return True
    return False

    #chinese char_based parser
            #
def char_zh_parser(string):
    if string == '':
        return string
    ans = ''
    for i in range(len(string)-1):
        ans += string[i]
        if zh(string[i]) or zh(string[i+1]) :
            ans += ' '
    ans += string[-1]
    return ans


def generate(batch_size, method):
    print('Generate')
    from gensim import models
    MODEL = models.Word2Vec.load("data/seg_data.model.bin")
    dict_ = {v:i+4 for i, v in enumerate(MODEL.wv.index2word)}
    def tokenizer(texts):
        def to_token(sen):
            sen = sen.replace('\n',' ')
            list_ = sen.split(' ')
            list_ = [i for i in list_ if i != ' ' and i!='']
            return [dict_[i] if i in MODEL.wv.index2word else 2 for i in list_]
        return [to_token(sentence[:50]) for sentence in texts]
    if method == 'without_idx' :
        '''
        generate batches of input attri from train.csv
        '''

        d = pd.read_csv('data/train.csv')

        d = d[pd.notnull(d.review)]
        #d['name']
        #d['u_score']
        nl = list(set(d['name']))
        key = {nl[i]:i for i in range(len(nl))}
        #d['id'] = [key[n] for n in d['name']]
        #d['name'] = [key[n] for n in d['name']]
        del d['name']
        d['review'] = [re.sub('[!,.]',' ',string) for string in d['review']]

        Seqs = tokenizer(d['review'])
        del d['review']
        d['Seqs'] = Seqs
        d['len'] = [len(sen) for sen in Seqs]
        d= d.sort_values(by=['len'])
        print('get word_index')
        embed()
        #yield tokenizer.word_index
        time = int(d.shape[0]/ batch_size)
        yield time
        while True :
            #d = d.sample(frac=1)
            for t in range(time) :
                #yield to_data(d.sample(n=batch_size).values)
                yield to_data(d[t*batch_size:(t+1)*batch_size].values)

def to_data(d):
    #del d['Unnamed: 0']
    #del d['u_score']
    x = []
    y = []
    embed()
    for row in d :
        x.append(row[0:9].astype(int))
        y.append(row[9]+[EOS])
    return (np.asarray(x), np.asarray(y))

#if __name__ == '__main__' :
