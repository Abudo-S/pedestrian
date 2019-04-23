from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
# from sklearn.model_selection import train_test_split

# from sklearn import svm
from SVM import SVM

'''
c=Image.open("img_00000.pgm")
print(list(c.getdata()))
'''
# 4800
length1 = 200
# 5000
length2 = 200

labels = []
# images_data =[]

test_labels = []
# images_test_data = []
features = []

for i in range(648):
    features.append("feat" + str(i))

df1 = pd.DataFrame(columns=features)
df2 = pd.DataFrame(columns=features)
# load_training_dataset
for i in range(1, 4):
    for j in range(length1):  # 4800
        file_dir = '/ped_examples/'
        strr = (str(i) + file_dir)

        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        img = img.flatten()
        df1.loc[j + ((i - 1) * length1)] = img
        # images_data=np.append(images_data, img)
        labels.append(1)  # pedestrian

for i in range(1, 4):
    for j in range(length2):  # 5000
        file_dir = '/non-ped_examples/'
        strr = (str(i) + file_dir)

        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        img = img.flatten()
        df1.loc[j + ((i - 1) * length2 + (3 * length1))] = img
        # images_data=np.append(images_data, img)
        labels.append(0)  # non-pedestrian

# load_testing_dataset
for i in range(1, 3):
    for j in range(length1):  # 4800
        file_dir = "/ped_examples/"
        strr = ('T' + str(i) + file_dir)

        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        img = img.flatten()
        df2.loc[j + ((i - 1) * length1)] = img
        # images_test_data=np.append(images_test_data, img)
        test_labels.append(1)  # pedestrian

    for j in range(length2):  # 5000
        file_dir = "/non-ped_examples/"
        strr = ('T' + str(i) + file_dir)

        img = np.array(Image.open(strr + "img_" + "{0:05}".format(j) + ".pgm"))
        img = img.flatten()
        df2.loc[j + ((i - 1) * length2 + (2 * length1))] = img
        # images_test_data=np.append(images_test_data, img)
        test_labels.append(0)  # non-pedestrian

print(df1)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(36, 18)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train = pd.read_csv('pedestrian.csv')
X1 = train.drop('ped',axis=1).values
y1 = train['ped'].values

test=pd.read_csv('test_pedestrian.csv')
X2 = test.drop('ped',axis=1).values
y2 = test['ped'].values
model.fit(X1, y1, epochs=5)
model.evaluate(X2, y2)

def cnn_model_fn(features, labels, mode):
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

cnn_model_fn(X1,y1,tf.estimator.ModeKeys.TRAIN)
cnn_model_fn(X1,y1,tf.estimator.ModeKeys.PREDICT)

