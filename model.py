import tensorflow as tf
keras=tf.keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras import Sequential
import os
from PIL import Image

outputs=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
dataset="MACHINELEARNING/HANDREC/DATASET"
contents=os.listdir(dataset)
# print(contents)

def set_index_to_one(lst, index):
    new_lst = [0 for _ in range(len(lst))]
    new_lst[index] = 1
    return new_lst

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for i in contents:
    # print(i)
    # pprint(os.listdir(f"{dataset}/{i}"))
    images=os.listdir(f"{dataset}/{i}")
    for j in images[0:20]:
        x_train.append(np.asarray(Image.open(f"{dataset}/{i}/{j}")))
        y_train.append(set_index_to_one(outputs,outputs.index(f"{i}")))
    for j in images[20:]:
        x_test.append(np.asarray(Image.open(f"{dataset}/{i}/{j}")))
        y_test.append(set_index_to_one(outputs,outputs.index(f"{i}")))

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

# print(outputs[np.argmax(y_test[-1])])
# for key,value in train.items():
#     for image in value:
#         img=f"{dataset}/{key}/{image}"
#         img=cv2.imread(img,0)
#         cv2.imshow(f'{key}', img)
#         cv2.waitKey(0)

model=Sequential()
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(300,300,3)))
model.add(MaxPooling2D((3,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(36,activation='relu'))

model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train,epochs=8,batch_size=30)

test_loss, test_acc = model.evaluate(x_test,  y_test)
print(f"Theaccuracy on test set is : {(test_acc*100):.6f}")

x=int(input("Choose an image between 1-720 : "))
while x!=0:
    image=x_train[x]
    plt.imshow(image, cmap='gray')
    num=model.predict(image.reshape(1,300,300,3))
    num=outputs[np.argmax(num)]
    plt.title(f"My neural network predicts {num} !")
    plt.show()
    x=int(input("Choose an image between 1-720: ")) 