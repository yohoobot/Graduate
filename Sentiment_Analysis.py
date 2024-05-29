# coding in colab
"-----------------------数据集准备---------------------------"
import nltk                   # Python的NLP库
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK

# download sample twitter dataset.
nltk.download('twitter_samples')

# 导入positive和negative推文 tweets load
origin_positive_tweets = twitter_samples.strings('positive_tweets.json')
origin_negative_tweets = twitter_samples.strings('negative_tweets.json')

# 查看数据集特性：
print('positive set type: ', type(origin_positive_tweets))
print('negative set type: ', type(origin_negative_tweets))

print('positive sample type: ', type(origin_positive_tweets[0]))
print('negative sample type: ', type(origin_negative_tweets[0]))

print('tweet num in positive dataset = ',len(origin_positive_tweets))
print('tweet num in negative dataset = ',len(origin_negative_tweets))


"-----------------------文本预处理---------------------------"
" 1.随机选取一条推文"
tweet = origin_positive_tweets[2040]
print(tweet)
tweet = 'RT '+tweet

" 2.去掉转发符 RT :转发的推文实例会在最前方加入RT字母"
#定义remove函数
import re #导入re模块，其中包含了正则表达式相关的方法，描述字符串特征更加准确

# 定义'RT '的remove函数
def remove_first_three(string):
  # ^元字符表示识别的是开始位置的'RT '，将原字符串中这类字符串替换为空串
  test = re.sub(r'^RT[\s]+', '', string)
  return test
print(tweet)

# 保留不在开头的'RT '
test2 = 'there is no RT '
test2 = remove_first_three(test2)
print(test2)

# test remove_first_three()
print('tweet: ', tweet)
tweet1 = remove_first_three(tweet)
print('tweet1:', tweet1)

# 如果检测不到开头的RT字符，则返回源字符串
print(tweet1)
print(remove_first_three(tweet1))

# 创建新列表，对10000条推文进行去RT操作并加入新列表。
processed_positive_tweets = []
processed_negative_tweets = []
for i in range(5000):
  processed_positive_tweets.append(remove_first_three(origin_positive_tweets[i]))
  processed_negative_tweets.append(remove_first_three(origin_negative_tweets[i]))
# test
print(origin_positive_tweets[2040])
print(processed_positive_tweets[2040])

"# 3.去掉超链接"
# 定义remove_hyperlinks函数
def remove_hyperlinks(text):
  clean_text = re.sub(r'https?://[^\s\n\r]+', '', text)
  return clean_text
# test
print('tweet1\n',tweet1)
tweet2 = remove_hyperlinks(tweet1)
print('tweet2:\n',tweet2)

# 对10000条推文进行去超链接操作并加入列表
for i in range(5000):
  processed_positive_tweets[i] = remove_hyperlinks(processed_positive_tweets[i])
  processed_negative_tweets[i] = remove_hyperlinks(processed_negative_tweets[i])
# test
i = 2040
print(origin_positive_tweets[i])
print(processed_positive_tweets[i])

" 4.去掉 # 号"
def remove_hashtags(text):
  clean_text = re.sub(r'#', '', text)
  return clean_text
# test
print('tweet2\n',tweet2)
tweet3 = remove_hashtags(tweet2)
print('tweet3:\n',tweet3)
# 对10000条推文进行去#操作并加入列表
for i in range(5000):
  processed_positive_tweets[i] = remove_hashtags(processed_positive_tweets[i])
  processed_negative_tweets[i] = remove_hashtags(processed_negative_tweets[i])
# test
i = 2040
print(origin_positive_tweets[i])
print(processed_positive_tweets[i])

" 5.分词处理 "
# 导入TweetTokenizer()用于根据空格分词，并将单词都转为小写，便于后期匹配
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
# test
print('tweet3:\n',tweet3)
tweet4 = tokenizer.tokenize(tweet3)
print('tweet4:\n',tweet4)

# 对所有数据进行分词处理
for i in range(5000):
  processed_positive_tweets[i] = tokenizer.tokenize(processed_positive_tweets[i])
  processed_negative_tweets[i] = tokenizer.tokenize(processed_negative_tweets[i])
# test
i = 2040
print(origin_positive_tweets[i])
print(processed_positive_tweets[i])

" 6.去掉和情绪感知关系不大的停用词，以及符号"
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

#导入英语停用词库 from NLTK
stopwords_english = stopwords.words('english')
print('Stop words: ', stopwords_english)
print('num of stopwords_english: ',len(stopwords_english),'\n')
# 符号
print('标点符号:', string.punctuation)
print('type:',type(string.punctuation))
print('num of punctuation: ',len(string.punctuation),'\n')

# 定义remove_stopword_and_punc()函数
def remove_stopword_and_punc(text):
  clean_text = []
  for word in text: # 遍历列表中每个词
    # 如果当前词不是停用词也不是符号
    if (word not in stopwords_english and word not in string.punctuation):
      clean_text.append(word)
  return clean_text
# test
print('tweet4:\n',tweet4)
tweet5 = remove_stopword_and_punc(tweet4)
print('tweet5:\n',tweet5)

# 对所有数据进行清洗操作
for i in range(5000):
  processed_positive_tweets[i] = remove_stopword_and_punc(processed_positive_tweets[i])
  processed_negative_tweets[i] = remove_stopword_and_punc(processed_negative_tweets[i])
# test
i = 2040
print(processed_positive_tweets[i])
print(origin_positive_tweets[i])

""" 7.词干提取（对词进行分类) """
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# 定义str_stem()函数
def str_stem(text):
  clean_text = []
  for word in text: # 遍历列表中每个词
    s = stemmer.stem(word)  # stemming word
    clean_text.append(s)
  return clean_text
# test
print('tweet5:\n',tweet5)
tweet6 = str_stem(tweet5)
print('tweet6:\n',tweet6)

# 对所有数据进行stem操作
for i in range(5000):
  processed_positive_tweets[i] = str_stem(processed_positive_tweets[i])
  processed_negative_tweets[i] = str_stem(processed_negative_tweets[i])
# test
i = 2040
print(processed_positive_tweets[i])
print(origin_positive_tweets[i])

" ----------------------初步NLP情感分析：基于字段检索--------------------- "
"1.合并列表"
tweets = processed_positive_tweets + processed_negative_tweets
print(len(tweets))

" 2.转为numpy数据集"
import numpy as np
# 用于标记总数据集中哪些是来自positive-1，哪些是来自negative-0
y = np.append(np.ones(5000), np.zeros(5000))

" 3.频率统计"
# "定义频率统计函数"
# 输入：推文及标签集j
# 输出：频率统计字典
def build_freqs(tweets, labels):
  freqs = {}
  # 遍历每一条推文
  for i in range(len(tweets)):
    tweet = tweets[i] #当前推文
    sentiment = labels[i] #当前推文的positive or negative
    # 遍历当前推文的每个词，记录频率
    for word in tweet:
      pair = (word, sentiment) # eg.('i',1)表示当前词是在positive推文中的单词'i'
      if pair in freqs: #如果频率表中已经创建
        freqs[pair] = freqs[pair] + 1 #'i'在positive句子中出现的频率+1
      else:
        freqs[pair] = 1
  return freqs

# "对10000条数据中的字词进行频率统计操作"
freqs = build_freqs(tweets, y)

" 4.计算每一句的正向/负向特征值"
# """定义函数"""
# 输入：推文列表和对应的字词正负向频率记录表
# 输出：记录每一条推文正负向特征值的列表
def features(tweets,freqs):
 X = np.zeros((len(tweets),2))
 for i in range(len(tweets)): # 遍历推文
   t = tweets[i]
   for w in t: # 遍历当前推文的每一个词
     if (w,1) in freqs: #如果当前词有正向的频率记录
       X[i,0] = X[i,0] + freqs[(w,1)] #加入句子的正向特征值
     if (w,0) in freqs:
       X[i,1] = X[i,1] + freqs[(w,0)]
 return X
X = features(tweets,freqs)
print(X)
# 经过nlp初步清洗得到的每条推文的正向和负向权重
print(X.shape)
# 源数据给出的正负向标签
print(y.shape)

"-------------------------------机器学习和逻辑回归运算--------------------------------"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report

plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel('positive feature',fontsize=20)
plt.ylabel('negative feature',fontsize=20)
plt.xlim(0,2000)
plt.ylim(0,2000)
plt.title('Scatter plot of features')

"""划分训练集和测试集"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=4)
scaler = preprocessing.StandardScaler()

"""训练"""
scaler.fit(X_train, y_train)

"""标准化处理"""
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""绘制决策边界"""
def decision_boundary():
  # 定义x,y坐标的范围，左边界为0，右边界为2000，步长为10
  xl,xr,dx = 0,2000,10
  yl,yr,dy = 0,2000,10
  # 生成均匀分布的数据
  xscat = np.arange(xl,xr,dx)
  yscat = np.arange(yl,yr,dy)
  # 生成可以进行矩阵运算的矩阵
  frst = np.ones((len(yscat),1))*xscat.reshape(1,len(xscat))
  secd = yscat.reshape(len(yscat),1)*np.ones((1,len(xscat)))
  frst = frst.reshape(-1)
  secd = secd.reshape(-1)
  # 合并为二维数组
  p_grid = np.column_stack((frst,secd))
  p_grid_scaled = scaler.transform(p_grid)
  # 模型预测
  f_grid = model.predict(p_grid_scaled)
  f_grid = f_grid.reshape((len(yscat),len(xscat)))
  plt.contour(xscat,yscat,f_grid,levels=[0.5])
  return

"""神经网络计算"""
model = 0

model = Sequential([
  # 添加一个具有两个节点的全连接层，并设置激活函数为relu
  Dense(2,activation='relu',input_shape=(2,)),
  # 添加一个具有一个节点的全连接层，并设置激活函数为sigmoid
  Dense(1,activation='sigmoid')
])

# 编译模型，损失函数是binary_crossentropy二元交叉熵
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 训练
model.fit(X_train_scaled,y_train,epochs=100,verbose=1)
# 存储每一步的损失值并画图显示
J_list = model.history.history['loss']
plt.plot(J_list)

"""绘制决策边界"""
plt.scatter(X[:,0],X[:,1],c=y)
decision_boundary()
plt.xlabel('Feature A (Positive)', fontsize=12)
plt.ylabel('Feature B (Negative)', fontsize=12)
plt.xlim(0,2000)
plt.ylim(0,2000)
plt.title('Decision Boundary')

# 计算交叉熵损失值
print('test:')
y_test_hat = model.predict(X_test_scaled)
print(bce(y_test.reshape(-1,1), y_test_hat).numpy()) # 计算交叉熵损失值
print('train:')
y_train_hat = model.predict(X_train_scaled)
print(bce(y_train.reshape(-1,1), y_train_hat).numpy())

# 对训练集进行测试，输出概率值大于0.5-->1，否则标记为0
print("Training Classification Report:")
y_train_cat = 1*(model.predict(X_train_scaled) > 0.5)
print(classification_report(y_train,y_train_cat)) # 打印报告

# 对测试集进行测试，输出概率值大于0.5-->1，否则标记为0
print("Testing Classification Report:")
y_test_cat = 1*(model.predict(X_test) > 0.5)
print(classification_report(y_test,y_test_cat))

"""
负向推文：准确率（Precision）=0.99，召回率（Recall）=1.00。即在所有被预测为情感表达负向的推文中，99%确实是来自negative类；在所有来自negative类的真正的推文中，100%被预测为负向推文。

正向推文：准确率（Precision）=1.00，召回率（Recall）=0.99。即在所有被预测为情感表达正向的推文中，100%确实是来自positive类；在所有来自positive类的真正的推文中，99%被预测为负向推文。
"""