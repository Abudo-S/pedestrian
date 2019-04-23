import tensorflow as tf

class CNN:
    
         
    def __init__(self,train,test,train_size,test_size):
        self.train_images = train.reshape((train_size, 36, 18, 1))
        self.test_images = test.reshape((test_size, 36, 18, 1))
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
    
    def create_model(self,train_labels):
        self.model=tf.keras.Sequential()
        #32 output_filters
        #(non-)pedestrian image shape= 36*18
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(36,18, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        #64 output_filters
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        self.model.fit(self.train_images, train_labels, epochs=7)
        
    def print_accuracy(self,y_test):
       test_loss, test_acc = self.model.evaluate(self.test_images, y_test)
       print("CNN_accuracy:"+str(test_acc))
    