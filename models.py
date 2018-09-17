from keras.models import Sequential
from keras.layers import merge,Conv2D,MaxPool2D,Activation,Dense
from keras.layers import Input
from keras.models import Model
def FeatureNetwork():
    #add feature Net
    inputShape = (28,28,1)
    models = Sequential()
    models.add(Conv2D(24,input_shape=inputShape,padding='valid',kernel_size=7,strides=1))
    models.add(Activation('relu'))
    models.add(MaxPool2D(strides=(2,2)))

    models.add(Conv2D(64,(5,5),strides=1,padding='valid'))
    models.add(Activation('relu'))
    
    models.add(Conv2D(96,kernel_size=3,strides=1,padding='valid'))
    models.add(Activation('relu'))

    models.add(Conv2D(96,kernel_size=3,strides=1,padding='valid'))
    models.add(Activation('relu'))

    models.add(Conv2D(64,kernel_size=3,strides=2,padding='valid'))
    models.add(Activation('relu'))
    return models

def ClassiFilerNet():#add classifier Net
    input1 = FeatureNetwork()
    input2 = FeatureNetwork()
    fci = merge([input1.output,input2.output])
    fc1 = Dense(1024,activation='relu')(fci)
    fc2 = Dense(1024,activation='relu')(fc1)
    fc3 = Dense(1024,activation='softmax')(fc2)
    models = Model(inputs=[input1.input,input2.input],outputs=fc3)
    return models

matchnet = ClassiFilerNet()
from keras.utils.vis_utils import plot_model
plot_model(matchnet, to_file='model.png')