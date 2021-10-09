#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from ast import literal_eval

train_data = pd.read_csv('./train_data_public.csv')
train_data.drop('Unnamed: 0', axis = 1, inplace = True)
test_data = pd.read_csv('./test_public.csv')

train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))
train_data['training_data'] = train_data.apply(lambda row: (list(row['text']), row['BIO_anno']), axis = 1)
test_data['testing_data'] = test_data.apply(lambda row: (list(row['text'])), axis = 1)
train_data


# In[2]:


training_data_txt = []
testing_data_txt = []

for i in range(len(train_data)):
    training_data_txt.append(train_data.iloc[i]['training_data'])
    
for i in range(len(test_data)):
    testing_data_txt.append(test_data.iloc[i]['testing_data'])


# ### Bi-LSTM Conditional Random Field
# ### pytorch tutorials https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

# In[3]:


import sys
import torch
print("Python version:  %s"%(sys.version))
print ("Torch version:  %s"%(torch.__version__))


# In[4]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


# #### 任务1：实体识别

# In[5]:


def argmax(vec):
    # return the argmax as a python int
    # 返回vec的dim为1维度上的最大值索引
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    # 将句子转化为ID
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
# 前向算法是不断累积之前的结果，这样就会有个缺点
# 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
# 为了避免这种情况，用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
# SUM = log(exp(s1)+exp(s2)+...+exp(s100))
#     = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
#     = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
# where clip=max
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# In[6]:


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim    # word embedding dim
        self.hidden_dim = hidden_dim          # Bi-LSTM hidden dim
        self.vocab_size = vocab_size          
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # 将BiLSTM提取的特征向量映射到特征空间，即经过全连接得到发射分数
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移矩阵的参数初始化，transitions[i,j]代表的是从第j个tag转移到第i个tag的转移分数
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START_TAG
        # 初始化STOP_TAG转移到所有其他tag的分数非常小，即不可能由STOP_TAG转移到其他tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 初始化LSTM的参数
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    
    def _get_lstm_features(self, sentence):
        # 通过Bi-LSTM提取特征
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        # 计算给定tag序列的分数，即一条路径的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 递推计算路径分数：转移分数 + 发射分数
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _forward_alg(self, feats):
        # 通过前向算法递推计算
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 初始化step 0即START位置的发射分数，START_TAG取0其他位置取-10000
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 将初始化START位置为0的发射分数赋值给previous
        previous = init_alphas

        # 迭代整个句子
        for obs in feats:
            # 当前时间步的前向tensor
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # 取出当前tag的发射分数，与之前时间步的tag无关
                emit_score = obs[next_tag].view(1, -1).expand(1, self.tagset_size)
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # 当前路径的分数：之前时间步分数 + 转移分数 + 发射分数
                next_tag_var = previous + trans_score + emit_score
                # 对当前分数取log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新previous 递推计算下一个时间步
            previous = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到STOP_TAG
        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 计算最终的分数
        scores = log_sum_exp(terminal_var)
        return scores


    def _viterbi_decode(self, feats):
        backpointers = []

        # 初始化viterbi的previous变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        previous = init_vvars
        for obs in feats:
            # 保存当前时间步的回溯指针
            bptrs_t = []
            # 保存当前时间步的viterbi变量
            viterbivars_t = []  

            for next_tag in range(self.tagset_size):
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                # 并不取决与当前tag的发射分数
                next_tag_var = previous + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 更新previous，加上当前tag的发射分数obs
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # CRF损失函数由两部分组成，真实路径的分数和所有路径的总分数。
        # 真实路径的分数应该是所有路径中分数最高的。
        # log真实路径的分数/log所有可能路径的分数，越大越好，构造crf loss函数取反，loss越小越好
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # 通过BiLSTM提取发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # 根据发射分数以及转移分数，通过viterbi解码找到一条最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# In[7]:


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 11
HIDDEN_DIM = 6
import time
t = time.time()

# 将训练集汉字使用数字表示
# 为了方便调试，先使用100条数据进行模型训练，选手可以采用全量数据进行训练
training_data = training_data_txt[:100] 
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# 将测试集汉字使用数字表示
testing_data = testing_data_txt
for sentence in testing_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = { "O": 0, "B-BANK": 1, "I-BANK": 2, "B-PRODUCT":3,'I-PRODUCT':4, 
             'B-COMMENTS_N':5, 'I-COMMENTS_N':6, 'B-COMMENTS_ADJ':7, 
             'I-COMMENTS_ADJ':8, START_TAG: 9, STOP_TAG: 10}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 训练前检查模型预测结果
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))
    a = model(precheck_sent)
    a = pd.Series(a)
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10): 
    print('the',epoch,' epoch')
    print(f'Time Taken: {round(time.time()-t)} seconds')
    for sentence, tags in training_data:
        # 第一步，pytorch梯度累积，需要清零梯度
        model.zero_grad()

        # 第二步，将输入转化为tensors
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 进行前向计算，取出crf loss
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 第四步，计算loss，梯度，通过optimier更新参数
        loss.backward()
        optimizer.step()

# 训练结束查看模型预测结果，对比观察模型是否学到
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[3][0], word_to_ix)
    print(model(precheck_sent))
    a = model(precheck_sent)
    a = pd.Series(a)
    a.to_csv('test1.csv')
# We got it!


# #### 任务2：情感分类
# 
# 1. 使用TF-IDF向量化
# 2. 线性回归进行分类

# In[8]:


X_train = list(train_data['text'])
X_test = list(train_data['text'][800:1000])

y_train = list(train_data['class'])
y_test = list(train_data['class'][800:1000])

test_data_sent = list(test_data['text']) # 列表格式
text_all = X_train + test_data_sent


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(text_all)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))


# In[10]:


X_train = vectoriser.transform(X_train)
test_data_text  = vectoriser.transform(test_data_sent)
# X_test  = vectoriser.transform(X_test)
print(f'Data Transformed.')


# In[11]:


from sklearn.linear_model import LogisticRegression

LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)


# In[12]:


import pickle
file = open('vectoriser-ngram-(1,2).pickle','wb')
pickle.dump(vectoriser, file)
file.close()

file = open('Sentiment-LR.pickle','wb')
pickle.dump(LRmodel, file)
file.close()


# In[13]:


def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(text)
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','class'])
    return df

sentiment_result = predict(vectoriser, LRmodel, X_test)
sentiment_result


# #### 生成提交文件

# In[14]:


test_data = pd.read_csv('./test_public.csv')
test_data


# In[16]:


ix_to_tag = dict([v,k] for k, v in tag_to_ix.items())


# ## 在test数据集预测结果

# In[18]:


# 实体识别结果
result = []
with torch.no_grad():
    for i in range(len(test_data)):
        precheck_sent = prepare_sequence(test_data.iloc[i][1], word_to_ix)
        sig_res = model(precheck_sent)[1]
        for i in range(len(sig_res)):
            sig_res[i] = ix_to_tag[sig_res[i]]
        result.append(' '.join(sig_res))
test_data['BIO_anno'] = result


# In[19]:


result[:10]


# In[20]:


# 预测test data 的sentiment 分类
sentiment_result = predict(vectoriser, LRmodel, test_data_sent)
test_data['class'] = list(sentiment_result['class'])
# test_data.to_csv('test_baseline.csv', index = None)

