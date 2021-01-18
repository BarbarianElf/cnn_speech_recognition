"""
cnn models file
for CNN Speech Recognition project

Created on January 12th 2021

@authors: Niv Ben Ami & Ziv Zango
"""
from keras import Sequential, losses, optimizers
from keras.activations import relu, softmax
from keras.layers import Input, BatchNormalization, Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping


def _save_model_plot(model, graphivz_path='C:/Program Files (x86)/Graphviz 2.44.1/bin'):
    """
    plot Keras model to png

    WARNINGS:
        please make sure the software graphviz is installed
        installation instruction: https://graphviz.org/download/
        WHEN INSTALL MAKE SURE ADDED GRAPHVIZ TO PATH
        change graphivz_path variable respectively
        -------
        please make sure the packages pydot, graphviz are installed
        install by run the command `pip install pydot graphviz`
        -------
        running on WINDOWS, please open PowerShell as Administrator
        run `dot -c`
    """
    try:
        from keras.utils.vis_utils import plot_model
        import os
        os.environ["PATH"] += os.pathsep + graphivz_path

        plot_model(model, to_file=f"results/{model.name}_model.png", show_shapes=True, show_layer_names=True)
    except ImportError as error:
        print(str(error) + "\nCAN'T SAVE PLOT MODEL")
        pass


def get_early_stop(value_monitored='val_loss', min_delta=0.01, num_of_epochs=10):
    return EarlyStopping(monitor=value_monitored, min_delta=min_delta, patience=num_of_epochs, verbose=1, mode='auto')


def get_model(model_type, *args):
    if model_type == 'mfcc':
        return get_mfcc_model(*args)
    if model_type == 'mel_spec':
        return get_mel_spec_model(*args)


def get_mfcc_model(input_shape, num_classes, learning_rate=0.001):
    """
    CNN model architecture for MFCC speaker recognition
    role model from the following article:
    http://www.cs.tut.fi/~tuomasv/papers/ijcnn_paper_valenti_extended.pdf

    """
    _name = 'mfcc'
    model = Sequential(name=_name)
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPool2D())

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPool2D())

    model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(relu))

    model.add(Dense(num_classes, activation=softmax))

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # Unmake the command below to save the model plot *READ WARNINGS*
    # _save_model_plot(model)

    # Unmark the command below to print the model summary
    # model.summary()

    return model


def get_mel_spec_model(input_shape, num_classes, learning_rate=0.001):
    """
    CNN model architecture for Mel-Spectrogram single word recognition
    role model from the following article:
    http://noiselab.ucsd.edu/ECE228_2019/Reports/Report38.pdf
    """
    _name = 'mel_spec'
    model = Sequential(name=_name)
    model.add(Input(shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(5, 5)))
    model.add(Activation(relu))
    model.add(Conv2D(32, kernel_size=(5, 5)))
    model.add(Activation(relu))
    model.add(MaxPool2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation(relu))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation(relu))
    model.add(MaxPool2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(Activation(relu))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Dense(128))
    model.add(Activation(relu))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation=softmax))

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # Unmake the command below to save the model plot *READ WARNINGS*
    # _save_model_plot(model)

    # Unmark the command below to print the model summary
    # model.summary()

    return model
