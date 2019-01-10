#coding=utf-8
'''
  使用LSTM的循环神经网络训练一个语言模型，给定上文语境，即历史出现的单词，
语言模型可以预测下一个单词出现的概率.训练中使用的数据集为PTB，训练集的一个batch示例如下：
   input:
    [[9970 9971 9972]
     [1969    0   98]]
   target:
    [[9975 9976 9980]
     [2254    0  312]]
  其中batchsize = 2，串联的LSTM的cell个数为 3，batch中的数字代表单词在词汇表中的唯一编号.
'''
import tensorflow as tf
import numpy as np
import time
import reader

# 定义语言模型处理输入数据的class,其中只有一个初始化方法__init__()
class PTBInput(object):
	def __init__(self, config, data, name = None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) //num_steps
		self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

# 定义LSTM的模型
class PTBModel(object):
	def __init__(self, is_training, config, input_):
		self._input = input_
		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		def lstm_cell():
			return tf.contrib.rnn.LSTMCell(size, forget_bias=0.0, state_is_tuple=True)
		
		attn_cell = lstm_cell
		if is_training and config.keep_prob < 1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = config.keep_prob)
		
		# 定义LSTM的一个cell
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
		
		# LSTM的一个cell的初始状态
		self._initial_state = cell.zero_state(batch_size, tf.float32)

		# 根据输入在词嵌入矩阵中查询出对应的词向量
		with tf.device('/cpu:0'):
			embedding = tf.get_variable('embedding', [vocab_size, size], dtype = tf.float32)
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
		
		# 对输入做Dropout处理
		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)
		
		# 搭建搭建LSTM网络，即一次RNN的一个batch一次正向传播过程
		outputs = []                      # 保存每个LSTM的cell的输入结果
		state = self._initial_state       # LSTM网络的初始状态，每跑一次batch时会feed到网络中
		with tf.variable_scope('RNN'):
			for time_step in range(num_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		
		# 计算训练损失
		output = tf.reshape(tf.concat(outputs, 1), [-1, size])
		softmax_w = tf.get_variable('softmax_w', [size, vocab_size], dtype = tf.float32)
		softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype = tf.float32 )
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			    [logits],
			    [tf.reshape(input_.targets, [-1])],
			    [tf.ones([batch_size*num_steps], dtype = tf.float32)])
		
		# 保存损失值
		self._cost = cost = tf.reduce_sum(loss)/batch_size
		
		# 保存最终的cell状态
		self._final_state = state

		# 如果不是训练，则直接返回，否则用优化器进行优化
		if not is_training:
			return 
		self._lr = tf.Variable(0.0, trainable = False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars), 
			             global_step = tf.train.get_or_create_global_step())
		self._new_lr = tf.placeholder(tf.float32, shape = [], name = 'new_learning_rate')
		self._lr_update = tf.assign(self._lr, self._new_lr)
	
	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict = {self._new_lr:lr_value})
	
	@property
	def input(self):
		return self._input
	
	@property
	def initial_state(self):
		return self._initial_state
	
	@property
	def cost(self):
		return self._cost
	
	@property
	def final_state(self):
		return self._final_state
	
	@property
	def lr(self):
		return self._lr
	
	@property
	def train_op(self):
		return self._train_op

# 小网络的配置
class SmallConfig(object):
	init_scale = 0.1           # 初始化器尺度范围
	learning_rate = 1.0        # 学习率大小
	max_grad_norm = 5          # 梯度裁剪中控制梯度最大范数的一个参数
	num_layers = 2             # LSTM的Cell的网络层数
	num_steps = 20             # LSTM的展开步数，即有多少个LSTM单元串到一起
	hidden_size = 200          # LSTM隐藏层的单元数，也即词向量的长度
	max_epoch = 4              # 初始学习率可训练的epoch数
	max_max_epoch = 13         # 总共可训练的epoch数
	keep_prob = 1.0            # dropout的保留概率
	lr_decay = 0.5             # 学习率衰减率大小
	batch_size = 20            # batch_size大小
	vocab_size = 10000         # 整个词汇表的大小

# 中等网络的配置
class MediumConfig(object):
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epoch = 6
	max_max_epoch = 39
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20
	vocab_size = 10000

# 大网络的配置
class LargeConfig(object):
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	lr_decay = 1/1.15
	batch_size = 20
	vocab_size = 10000

# 测试集的配置
class TestConfig(object):
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1
	num_layers = 1
	num_steps = 2
	hidden_size = 2
	max_epoch = 1
	max_max_epoch = 1
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

# 运行一次epoch的流程
def run_epoch(session, model, eval_op = None, verbose = False):
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model._initial_state)

	fetches = {'cost': model.cost, 'final_state':model.final_state}
	if eval_op is not None:
		fetches['eval_op'] = eval_op

	# 每次跑一个batch，每跑一个batch时, 初始state中的(c,h)都要用初始化后的值来feed到网络中
	for step in range(model.input.epoch_size):
		feed_dict = {}

		# 将一个LSTM单元的全部初始化后的state加入feed_dict中
		for i, (c,h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h
		vals = session.run(fetches, feed_dict)
		cost = vals['cost']
		state = vals['final_state']

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size //10) == 10:
			print ('%.3f perplexity: %.3f speed: %.0f wps' %
				  (step*1.0/model.input.epoch_size, np.exp(costs/iters),
				  	iters*model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs/iters)

# 开始训练
def main():

	# 获取原始单词数据，这里的data均为数字编码序列 train:929589, valid:73760, test:82430
	train_data, valid_data, test_data, _ = reader.ptb_raw_data('./simple-examples/data/')
	
	# 获取训练过程中的各种配置
	config = SmallConfig()
	eval_config = SmallConfig()
	eval_config.batch_size = 2   # 设置batch_size大小
	eval_config.num_steps = 2    # 设置LSTM的展开步骤，即多少个cell串联在一起

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
		
		# 准备训练集和RNN模型
		with tf.name_scope('train'):
			train_input = PTBInput(config = config, data = train_data, name = 'TrainInput')
			with tf.variable_scope('Model', reuse = None, initializer = initializer):
				m = PTBModel(is_training = True, config = config, input_ = train_input)
		
		# 准备验证集和RNN模型
		with tf.name_scope('Valid'):
			valid_input = PTBInput(config = config, data = valid_data, name = 'ValidInput')
			with tf.variable_scope('Model', reuse = True, initializer = initializer):
				mvalid = PTBModel(is_training = False, config = config, input_ = valid_input)
		
		# 准备测试集和RNN模型
		with tf.name_scope('Test'):
			test_input = PTBInput(config = config, data = test_data, name = 'TestInput')
			with tf.variable_scope('Model', reuse = True, initializer = initializer):
				mtest = PTBModel(is_training = False, config = eval_config, input_ = test_input)

		sv = tf.train.Supervisor()
		with sv.managed_session() as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay**max(i + 1 - config.max_epoch, 0.0)
				m.assign_lr = (session, config.learning_rate*lr_decay)

				print ('epoch: %d Learning_rate: %.3f' % (i + 1, session.run(m.lr)))

				train_perplexity = run_epoch(session, m, eval_op = m.train_op, verbose = True)
				print ('epoch: %d Train Perplexity: %.3f' %(i+1, train_perplexity))

				valid_perplexity = run_epoch(session, mvalid)
				print ('epoch: %d Valid Perplexity: %.3f' %(i+1, valid_perplexity))

		test_perplexity = run_epoch(session, mtest)
		print ('Test Perplexity: %.3f' % test_perplexity)

def test():
	# train:929589, valid:73760, test:82430
	train_data, valid_data, test_data, _ = reader.ptb_raw_data('./simple-examples/data/')
	input_data, targets = reader.ptb_producer(train_data, batch_size = 2, num_steps = 3)
	print ('train_data:',len(train_data),'val_data:',len(valid_data),'test_data:',len(test_data))
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		
		inputs = sess.run(input_data)
		target = sess.run(targets)
		print ('input:')		
		print (inputs)
		print ('target:')
		print (target)
		
		coord.request_stop()
		coord.join(threads)


if __name__ == '__main__':
	main()
	# test()