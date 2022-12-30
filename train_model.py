from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
import pandas as pd
import tensorflowjs as tfjs

glcm = pd.read_csv("/content/gdrive/MyDrive/DATASET/CSV/TrainGLCM.csv", index_col = 0)

# Separate Feature and Target Matrix and change it into array format
X = glcm.drop('label', axis = 1).values
y = glcm.label.values
print(X.shape)

x = X.reshape(X.shape[0], X.shape[1], 1)
print(x.shape)

print(unique(y))
print(unique(y).sum())

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2)

#----------------------------------------------------------------

model = Sequential()
model.add(Conv1D(64, 3, padding='same', activation="relu", input_shape=(24,1)))

#Second CNN layer  with 64 filters, conv window 3, relu activation and same padding
model.add(Conv1D(64, 3, padding='same', activation="relu"))

#Third CNN layer with 128 filters, conv window 3, relu activation and same padding
model.add(Conv1D(128, 3, padding='same', activation="relu"))

#Fourth CNN layer with Max pooling
model.add(MaxPooling1D(pool_size=(3,), strides=2, padding='same'))
#model.add(Dropout(0.5)) #to prevent overfitting

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512, activation="relu"))
model.add(Flatten())
#model.add(Dropout(0.3))

model.add(Dense(30, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])

model.summary()

#----------------------------------

# Set a learning rate annealer
from keras.callbacks import ReduceLROnPlateau #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

history = model.fit(xtrain, ytrain, batch_size=500, epochs=100, validation_data = (xtest, ytest), callbacks=[learning_rate_reduction])

acc = model.evaluate(xtrain, ytrain)
print("Loss:", acc[0], " Training  Accuracy:", acc[1])

acc = model.evaluate(xtest, ytest)
print("Loss:", acc[0], " Testing Accuracy:", acc[1])

pred = model.predict(xtest)
pred_y = pred.argmax(axis=-1)

cm = confusion_matrix(ytest, pred_y)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cm,
                     index = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing"], 
                     columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing"])
#Plotting the confusion matrix
plt.figure(figsize=(20,20))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

import numpy as np
preds = np.round(model.predict(xtest), 0)

alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing"]

from sklearn import metrics
from sklearn.metrics import confusion_matrix
classification_metrics = metrics.classification_report(ytest, pred_y)
print(classification_metrics)


#===========Plotting the Accuracy============================
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.title('Accuracy Vs Val_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

tfjs.converters.save_keras_model(model, "model_jss")
