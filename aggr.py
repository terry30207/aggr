import os
import re
import numpy
import tensorflow
from functools import reduce
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

numpy.random.seed(7)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255
    X_test = X_test / 255
    model_list=[]
    weight_list=[]
    #Reload model
    for root, dirs, files in os.walk("..", topdown=False):
        for name in files:
            path=os.path.join(root, name)
            print(path)
            if re.search(r'.*h5$',path) != None:
                if re.search("global.h5",path)==None:
                    model_list.append(path)

    #!!!!!GET WEIGHT OUT!!!!!
    for mod_path in model_list:
        mod=load_model(mod_path)
        weight_list.append(numpy.array(mod.get_weights()))

    #Compute avg

    weight=reduce(numpy.add,weight_list)
    weight=weight/float(len(weight_list))

    tmodel=load_model(model_list[0])
    scores = tmodel.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    #!!!!!PUT WEIGHT BACK!!!!!
    model=load_model("global.h5")
    model.set_weights(weight)

    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.save("global.h5")