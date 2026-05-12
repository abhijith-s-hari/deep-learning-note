import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Sequential


class cuslayer(Layer):
    def __init__(self, units = 16) :
    # unit is the no of neurons the layer will have when the function is called
    # if the no of neuron per layer is not passed at the time of calling 
        super(cuslayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape[-1], self.units), 
                                 trainable=True, 
                                 initializer='random_normal')
        self.b = self.add_weight(shape = (self.units, ), 
                                 trainable=True, 
                                 initializer="zeros")
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
    

model = Sequential([
    cuslayer(64), 
    cuslayer(32),
    Softmax()
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.build((1000, 20))
model.summary()

