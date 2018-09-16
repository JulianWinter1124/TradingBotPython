import os



class Neural():

    def __init__(self, model_name, overwrite, batch_size, output_size, n_in, n_out, n_features, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='mse', optimizer='adam'):
        self.optimizer = optimizer #Keras has many optimizers, I will use adam
        self.loss_function = loss_function
        self.n_out = n_out #n_out + 1 points are getting predicted
        self.activation_function = activation_function
        self.layer_units = layer_units #list of how many neurons each layer has
        self.n_in = n_in #Data prior to present
        self.n_features = n_features #the number of columns the input data has
        from keras.models import Sequential
        from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        self.overwrite = overwrite #overwrite saved models?
        self.batch_size = batch_size  # how much data is processed at once
        self.filepath = 'data/training/LSTM_' + model_name + '.hdf5' # Model is saved here
        self.output_size = output_size #the output size should be n_out + 1
        self.model: Sequential = None
        # From https://www.kaggle.com/cbryant/keras-cnn-with-pseudolabeling-0-1514-lb/ might need tuning
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min') #If there are 10 epochs without progression in val_loss, stop the training
        self.mcp_save = ModelCheckpoint(self.filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1) # Saves the model when there is a better one

    def load_or_build_model(self):
        """
        Loads the model from file or creates one if not existent
        """
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, Flatten
        if self.overwrite or not os.path.isfile(self.filepath):  # Is there no existing model?
            self.model = Sequential() #Start by using this
            for i in range(0, len(self.layer_units)):
                self.model.add(
                    LSTM(units=self.layer_units[i], input_shape=(self.n_in, self.n_features), return_sequences=True)) #return_sequences=shape is passed along and not altered
                if self.activation_function == 'LeakyReLU':
                    self.model.add(LeakyReLU(alpha=.001)) #Keras has no string shortcut in Activation for LeakyReLu
                else:
                    self.model.add(Activation(self.activation_function)) #add activation
                self.model.add(Dropout(0.5)) #How many Neurons should be discarded? 0.5 = 50%
            self.model.add(Flatten()) #flattens the output shape so it fits in a Dense block
            self.model.add(Dense(self.output_size))
            self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
            self.model.save(self.filepath, True)
        else:
            self.model = load_model(self.filepath)
        self.model.summary()

    def train_model(self, train_X, train_Y, test_X, test_Y, epochs, shuffle=True, save=True):
        """
        Trains the model with the given data
        :param train_X: The training data
        :param train_Y: The training label
        :param test_X: The test data
        :param test_Y: The test label
        :param epochs: The number of epochs the model is trained at max
        :param shuffle: if the data should be shuffled
        :param save: should progress be saved? almost always True
        :return: the training history containing val_loss, loss and other metrics
        """
        if save:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_split=0.1,
                                     epochs=epochs, callbacks=[self.mcp_save, self.earlyStopping],
                                     shuffle=shuffle)
        else:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_split=0.1,
                                     epochs=epochs, callbacks=[self.earlyStopping],
                                     shuffle=shuffle)
            self.model.evaluate(test_X, test_Y, batch_size=self.batch_size)
        return history


    def train_model_generator(self, generator, steps_per_epoch, epochs, use_multiprocessing=True, workers=4): #this is not used right now
        history = self.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=epochs, use_multiprocessing=use_multiprocessing, workers=workers, callbacks=[self.mcp_save, self.earlyStopping])
        return history

    def plot_model_to_file(self, filename, show_shapes, show_layer_names):
        from keras.utils import plot_model
        print('Plotting model to', filename)
        plot_model(self.model, filename, show_shapes, show_layer_names)

    def predict(self, data):
        return self.model.predict(data)

    def predict_on_batch(self, data): #not used right now
        return self.model.predict_on_batch(data)
