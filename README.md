# Session-5

In this assignment, We will include following for  mnist prediction and observe if it can improve the prediction  :-
  image normalization
  L2 regularization
  ReLU after BN
  
 Please note :- BatchNormalization, Learning rate scheduler, Drop Out are allready part of architecture( refer Session -5 Code 8).
 
 
## 1) image normalization
 
 We calculate mean and standard deviation of training data set and use this to noramlize both training and testing data.
 
           # create generator that centers pixel values
          datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
          # calculate the mean on the training dataset
          datagen.fit(X_train)
          # normalize entire train data set
          iterator = datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False)
          X_train, y_train = iterator.next()
          # normalize entire test data set
          iterator = datagen.flow(X_test, y_test, batch_size=len(X_test), shuffle=False)
          X_test, y_test = iterator.next()



## 2) L2 regularization

        I have tried to create a custom loss with l2 . 

        import keras.backend as K
        def reg_term(lambd):
          w2 = 0
          for i in range(len(model.layers)):
            if len(model.layers[i].get_weights()) >0:
              t = np.sum(model.layers[i].get_weights()[0]*model.layers[i].get_weights()[0])
              w2=w2+t
          w2 = w2*(lambd/(2*X_train.shape[0]))
          return w2
        def loss_l2(y_true,y_pred):
                return K.categorical_crossentropy(y_true, y_pred)+reg_term(lambd)
                
                
## 3) ReLU after BN

Ran the new architecture for 40 epochs and save the model with highest validation accuracy


### After making above changes and running the new architecture for 40 epochs, validation accuracy reaches to 99.59%  (in 40th epoch).

## Finally the image gallery for 25 misclassified image is cread.

Each image is named as 'index-actual-predicted'.


#### Final Observation - For most the images, the labelling seems to be incorrect(manual checking).
