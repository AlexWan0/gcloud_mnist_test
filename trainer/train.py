import tensorflow as tf
import preprocess

num_classes = 10
num_epochs = 10

height = 28
width = 28
num_channels = 1

batch_size = 128

def net(x):
	# reshape tensor to 4d where [batch size, height, width, channel]
	x = tf.reshape(x, shape=[-1, height, width, num_channels])

	# convolutional layer with 32 filters, kernel size 5, and relu activation
	conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
	# max pooling layer with stride 2 and kernel size 2
	conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

	# conv with 64 filters, kernel size 3, relu activation
	conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
	# max pooling with stride 2, kernel size 2
	conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

	# flatten to 1d vector
	flt = tf.contrib.layers.flatten(conv2)

	# dense layer of size 1024
	d1 = tf.layers.dense(flt, 1024)
	# adds dropout of 0.25
	d1 = tf.layers.dropout(d1, rate=0.25)

	# output layer
	out = tf.layers.dense(d1, num_classes)

	return out

#sets placeholder for input and output
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

predict_y = net(x)


# --training--

loss = tf.losses.mean_squared_error(labels=y, predictions=predict_y)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# load data
train_x, train_y = preprocess.train_data()

sess = tf.Session()

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(num_epochs):
	#total loss for epoch
	epoch_loss = 0

	#goes through each batch to train
	for i in range(batch_size, len(train_x), batch_size):
		print("\rTraining",str(i)+"/"+str(len(train_x)), end=" ")
		
		#gets current batch
		batch_x = train_x[i-batch_size:i]
		batch_y = train_y[i-batch_size:i]
		
		#runs optimizer and gets loss
		_, l = sess.run([optimizer, loss], feed_dict={x: batch_x, y:batch_y})
		
		#adds batch loss to epoch loss
		epoch_loss += l

	print("Epoch #", epoch, "loss:", epoch_loss)


# --testing--

# gets test data
test_x, test_y = preprocess.test_data()

correct = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

print('Accuracy:',accuracy.eval({x:test_x[0:1024], y:test_y[0:1024]},session=sess))