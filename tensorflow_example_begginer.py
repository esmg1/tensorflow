import tensorflow as tf
from tensorflow.keras import optimizers
tf.keras.backend.set_floatx('float64') #Arregla error con el tipo de float
mnist = tf.keras.datasets.mnist #Importa el dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data() #Divide el dataset en training y test data
x_train, x_test = x_train / 255.0, x_test / 255.0 #Ni idea. Lo normaliza?

model = tf.keras.models.Sequential([                #create the model layer by layer
  tf.keras.layers.Flatten(input_shape=(28, 28)),    #first layer, 28*28 input pixels
  tf.keras.layers.Dense(128, activation='relu'),    #second layer 128 hidden nodes
  tf.keras.layers.Dropout(0.2),                     #rate at which random dropout occurs (prevents overfitting)
  tf.keras.layers.Dense(10)                         #final layer, 0-9 digit output
])                                                  #se puede aplicar un activation='softmax'
                                                    #pero es preferible hacerlo después (línea 17)
                                                    
predictions = model(x_train[:1]).numpy()            #Vector entrada para softmax, que generará probabilidades normalizadas
tf.nn.softmax(predictions).numpy()                  #Probabilities for each class

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                                                    #takes a vector of logits and a true index and returns a scalor loss for each example
                                                    #equal to the negative log probability of the true class: It is zero if the model
                                                    #is sure of the correct class
                                                    #the initial loss should be close to -tf.log(1/10) ~=2.3
                                                    #p.d. loss function = cost function

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
                                                    #Constructing the model
                                                
model.fit(x_train, y_train, epochs=10)              #adjust parameters to minimize loss

model.evaluate(x_test,  y_test, verbose=2)          #checks the models performance

#can see probabilities by wrapping the model with a new softmax layer
# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
# probability_model(x_test[:5])


#save model
model.save('mnist.h5')

#load model
model = tf.keras.models.load_model('mnist.h5')



#----- you can test with new images ------
img = tf.keras.preprocessing.image.load_img('data/digit.png', color_mode = "grayscale", target_size=(28, 28))
img = tf.keras.preprocessing.image.img_to_array(img)
img = img.reshape(1,28,28)
img = img.astype('float64') / 255.0

model.predict_classes(img)
