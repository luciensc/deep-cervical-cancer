# Author: Lucien Schläpfer
# Date: last updated 20.08.2020
# works with keras 2.3.1, tensorflow 2.0.0

import argparse
import numpy as np
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model_defs import VGG_mod, R50_mod
from dataloaders import Data


parser = argparse.ArgumentParser('DCC')
parser.add_argument("--model", type=str, choices=["vgg", "resnet", "combined"], default="combined",
                    help="specify which model to use")
parser.add_argument("--data_path", type=str, default="/cluster/home/luciensc/sipakmed_formatted/",
                    help="path to downloaded and formatted SiPaKMed dataset folder.")
###########################################################################################################

#args = parser.parse_args("--model combined --data_path sipakmed_formatted/".split())
args = parser.parse_args()

###########################################################################################################
### loading data & base models
data = Data(path=args.data_path)

base_models = []
if args.model == "vgg":
    base_models.append(VGG_mod(42))
elif args.model == "resnet":
    base_models.append(R50_mod(42))
else: # both models
    base_models.append(VGG_mod(42))
    base_models.append(R50_mod(42))

###########################################################################################################
### training base models
epochs = 50

# define callback. purpose: restore best weights -> patience == epochs
restore_callback = EarlyStopping(monitor='val_acc', patience=epochs, restore_best_weights=True, verbose=2)
np.random.seed(0)  # seed training process
for model in base_models:
    # fast learning for 50 epochs
    opt = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    model.fit(data.train_set_DA, epochs=epochs, validation_data=data.val_set, verbose=2,
                        callbacks=[restore_callback])

    # fine-tuning for 50 epochs. building on best model in previous round based on val_acc. restore to best weights.
    opt = Adam(learning_rate=1e-5)
    model.fit(data.train_set_DA, epochs=epochs, validation_data=data.val_set, verbose=2,
                        callbacks=[restore_callback])

    # evaluate performance of base model
    y_test = data.test_set.classes
    pred = np.argmax(model.predict_generator(data.test_set, steps=data.test_set.n), axis=1)
    print("base model accuracy: ", accuracy_score(y_test, pred))
    print("base model confusion matrix: ", confusion_matrix(y_test, pred))
    print("base model classification report: ", classification_report(y_test, pred))

    if len(base_models)==1:  # i.e. if only evaluating a single base model, not the combination model
        exit(1)

###########################################################################################################
### continue only with the combination model
# creating the model extracting features from the last layer before the softmax layer.
vgg_extractor = Model(inputs=base_models[0].input, outputs=base_models[0].get_layer("dense_1024").output)
r50_extractor = Model(inputs=base_models[1].input, outputs=base_models[1].get_layer("dense_1024").output)

# perform feature extraction on data -> construct new dataset consisting of extracted features (2x1024), X; and labels y
dataFE = {}
for nm, dat in [("train", data.train_set_NO_DA), ("val", data.val_set), ("test", data.test_set)]:
    dataFE["y_"+nm] = to_categorical(dat.classes)
    vggpred = vgg_extractor.predict(dat)
    r50pred = r50_extractor.predict(dat)
    dataFE["X_"+nm] = np.concatenate([vggpred, r50pred], axis=1)

# train combination model:
opt = Adam(learning_rate=1e-3)
model = Sequential()
model.add(Dropout(0.75, input_shape=(2048,)))
model.add(BatchNormalization())
model.add(Dense(5, activation="softmax"))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

epochs = 200
restore_callback = EarlyStopping(monitor='val_acc', patience=epochs, restore_best_weights=True, verbose=2)  # overwrite previous cb
model.fit(dataFE["X_train"], dataFE["y_train"], batch_size=32, epochs=epochs, verbose=2, shuffle=True,
          validation_data=(dataFE["X_val"], dataFE["y_val"]), callbacks=[restore_callback])

# evaluate on test data
y_test = np.argmax(dataFE["y_test"], axis=1)  # transform from one hot encoded to class label
pred = np.argmax(model.predict(dataFE["X_test"]), axis=1)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


