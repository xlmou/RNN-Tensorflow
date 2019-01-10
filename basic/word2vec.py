#coding=utf-8
'''
  训练Word2Vec有两种模式，其中CBOW是从原始语句中推测目标字词，而Skip-Gram则正好相反，
它是从目标字词推测出原始语句. 这里主要采用第二种方式，即使用Skip-Gram模式的Word2Vec.
  一个训练batch的实例如下：
  batch : [3081 3081   12   12    6    6  195  195]
  label : [[5234] [12] [ 6] [3081] [195] [12] [2] [6]]
  这里的batchsize=8，数字代表单词在词汇表中的编号
'''
import collections
import numpy as np
import zipfile
import os 
import random
import math
import tensorflow as tf

vocabulary_size = 50000    # 只保存50000个词汇
batch_size = 128           # batch_size 大小
embedding_size = 128       # 词向量的长度
skip_window = 1            # 单词最远可以联系的距离
num_skips = 2              # 每个单词生成多少个样本

valid_size = 16            # 抽取验证的单词数
valid_window = 100         # 验证单词从频数最高的100个单词抽取
valid_examples = np.random.choice(valid_window, valid_size, replace = True)
num_sampled = 64           # 负样本噪声单词的数量
data_index = 0             # 单词序号


# 读取原始数据，得到一个单词列表
def read_data():
	with open('./text8/text8', 'r') as f:
		data = f.read()	
		data = data.split(' ')
		data = data[1:]	
	return data


# 根据原始的单词列表准备好可供训练的数据
def build_dataset():

	# count保存前50000个高频词的出现次数
	words = read_data()
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	
	# dictionary保存count中的(word, word在count中的位置)的键值对
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	
	# data保存words中的词在count中的位置
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count +=1
		data.append(index)

	# 未知的单词统计	
	count[0][1] = unk_count

	# 将字典的键值对进行翻转
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	
	return data, count, dictionary, reverse_dictionary

# 创建数据集，并保存为全局变量
data, count, dictionary, reverse_dictionary = build_dataset()
'''
   count : 列表，保存前50000个高频词的出现次数
   dictionary : 字典，保存count中的(word, word在count中的位置)的键值对
   data : 保存words中的词在count中的位置
   reverse_dictionary : dictionary的键值对的反转
'''


def generate_batch(batch_size, num_skips, skip_window):
	'''
	  生成训练中的batch数据
	
	参数介绍：
	  batch_size : batch的大小，它必须是num_skips的整数倍(确保一个batch包含了一个词汇对应的
	               所有样本)；
	  num_skips:   为每个单词生成多少个样本，它不能大于skip_window的两倍值；
	  skip_window: 单词最远可以联系的距离，设置为1表示只能跟紧邻的单词生成样本.比如：
	               'the quick brown'这三个单词中，quick只能跟前后的单词生成两个样本，
	               (quick, the)和 (quick, brown),样本中第一个词为train，第二个词为label
	
	返回值：
	   组装好的 batch和 labels
	'''
	global data_index
	assert batch_size % num_skips ==0
	assert num_skips <= 2*skip_window
	batch = np.ndarray(shape = (batch_size), dtype = np.int32)
	labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen = span)
	
	# 取出span个单词，保存其在count中的位置于buffer中
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	# 从buffer中生成batch和label，每次循环对一个目标单词生成多个样本
	for i in range(batch_size//num_skips):
		target = skip_window
		targets_to_avoid = [ skip_window ]

		# 这里的每次循环生成同一个目标单词的一个样本
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		
		# buffer的滑窗向后移动一位，这样我们的目标单词也向后移动一位
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels


def test():
	'''
	   模块测试
	'''
	words = read_data()
	print (len(words))       # 原始语料库中有17005207个单词
	print (words[:10])
	data, count, dictionary, reverse_dictionary = build_dataset()
	print (count[:5])                                  # 前5个出现次数最多的词
	print (data[:10])                                  # 前10个词dictionary中的编码，编码越小次数越多     
	print ([reverse_dictionary[i] for i in data[:10]]) # 前10个词
	batch, labels = generate_batch(batch_size = 8, num_skips = 2, skip_window = 1)
	print (batch)
	print (labels)
	for i in range(8):
		print (batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


def main():
	'''
	   开始训练
	'''
	graph = tf.Graph()
	with graph.as_default():
		train_inputs= tf.placeholder(tf.int32, shape = [batch_size])
		train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
		valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

		with tf.device('/cpu:0'):
			
			# 构建词向量矩阵
			embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1.0))
			
			# 根据输入batch在词向量矩阵中找到相应的词向量
			embed = tf.nn.embedding_lookup(embeddings, train_inputs)
			
			# 定义NCE损失的权重和偏置
			nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], 
				                      stddev = 1.0/math.sqrt(embedding_size)))
			nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
		
		# 计算NCE损失和定义优化器
		loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
											 biases = nce_biases,
											 labels = train_labels,
											 inputs = embed,
											 num_sampled = num_sampled,
											 num_classes = vocabulary_size))
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
		
		# 计算嵌入向量的L2范数并将嵌入向量除以二范数得到标准化后的嵌入向量
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
		normolized_embeddings = embeddings/norm
		
		# 查询验证单词的嵌入向量并计算验证单词的嵌入向量和词汇表中所有单词的相似性
		valid_embeddings = tf.nn.embedding_lookup(normolized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normolized_embeddings, transpose_b = True)
		
		init = tf.global_variables_initializer()	
		
		# 迭代总数
		num_steps = 100001

		with tf.Session() as sess:
			init.run()
			print ('initialized')
			average_loss = 0
			for step in range(num_steps):
				batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
				feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
				_, loss_val = sess.run([optimizer, loss], feed_dict = feed_dict)
				average_loss += loss_val

				# 每2000次循环，计算一次平均loss并显示出来
				if step%2000==0:
					if step > 0:
						average_loss /=2000
					print ('average_loss at step', step, ':', average_loss)
					average_loss = 0
				
				#  每10000次循环，计算一次验证单词和全部单词的相似度，
				# 并将与每个验证单词最相似的8个单词显示出来
				if step%10000==0:
					sim = similarity.eval()
					for i in range(valid_size):
						valid_word = reverse_dictionary[valid_examples[i]]
						top_k = 8
						nearest = (-sim[i, :]).argsort()[1:top_k + 1]
						log_str = 'nearest to %s:' % valid_word
						for k in range(top_k):
							close_word = reverse_dictionary[nearest[k]]
							log_str = '%s %s,'%(log_str, close_word)
						print (log_str)
			final_embeddings = normolized_embeddings.eval()


if __name__ == '__main__':
	main()
	# test()