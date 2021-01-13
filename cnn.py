
from keras import Sequential, losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import Dense, Dropout, Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten
from keras.callbacks import EarlyStopping


def _save_model_plot(model):
    try:
        from keras.utils.vis_utils import plot_model
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz 2.44.1/bin'

        plot_model(model, to_file=model.name + '_model.png', show_shapes=True, show_layer_names=True)
    except ImportError as error:
        print(str(error) + "\nCAN'T SAVE PLOT MODEL")
        pass


def get_early_stop(value_monitored='val_loss', min_delta=0.01, num_of_epochs=5):
    return EarlyStopping(monitor=value_monitored, min_delta=min_delta, patience=num_of_epochs, verbose=1, mode='auto')


def get_mfcc_model(input_shape, num_classes, learning_rate=0.001):
    _name = 'mfcc'
    model = Sequential(name=_name)
    model.add(Input(shape=input_shape))

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