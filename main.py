"""
main file
for CNN Speech Recognition project

Created on January 12th 2021

@authors: Niv Ben Ami & Ziv Zango
"""
from keras.models import load_model
import os
import numpy

import data_utils
import config
import cnn


def get_data(feature_type=None):
    try:
        x_data = numpy.load("saved_data/x_data_" + feature_type + ".npy")
        y_data = numpy.load("saved_data/y_data_" + feature_type + ".npy")
        x_test = numpy.load("saved_data/x_test_" + feature_type + ".npy")
        y_test = numpy.load("saved_data/y_test_" + feature_type + ".npy")
    except IOError:
        x_data, x_test, y_data, y_test = data_utils.process_feature(config.FREQUENCY_SAMPLED,
                                                                    config.FRAME_MAX_LEN,
                                                                    feature_type=feature_type,
                                                                    num_for_test=config.NUM_OF_TEST)
        x_data, x_test = data_utils.expand_dim(x_data, x_test)
        # saving to data to files for the next time about 500MB
        numpy.save("saved_data/x_data_" + feature_type + ".npy", x_data)
        numpy.save("saved_data/x_test_" + feature_type + ".npy", x_test)
        numpy.save("saved_data/y_data_" + feature_type + ".npy", y_data)
        numpy.save("saved_data/y_test_" + feature_type + ".npy", y_test)
    except Exception as err:
        raise Exception(f"Cant read data\n{err}")
    y_data, y_test, _encoder = data_utils.data_encoder_and_categorized(y_data, y_test)
    x_train, x_valid, y_train, y_valid = data_utils.train_valid_split(x_data, y_data, percent=config.VALID_PERCENT)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, _encoder


def _check_feature_input(feature_type):
    if feature_type == 'mfcc':
        return config.NUM_OF_SPEAKERS
    elif feature_type == "mel_spec":
        return config.NUM_OF_WORDS
    else:
        raise ValueError('feature_type must be `mfcc` or `mel_spec`')


def plot_training_many_batch(feature_type, batch_sizes):
    history = []
    # graph = ['loss', 'val_loss']
    graph = ['accuracy', 'val_accuracy']
    num_classes = _check_feature_input(feature_type)
    in_train, out_train, in_valid, out_valid, in_test, out_test, encoder = get_data(feature_type)
    for batch_size in batch_sizes:
        model = cnn.get_mfcc_model(in_train.shape[1:], num_classes, learning_rate=0.001)
        history.append(model.fit(in_train, out_train,
                                 batch_size=batch_size,
                                 epochs=100,
                                 validation_data=(in_valid, out_valid),
                                 callbacks=[cnn.get_early_stop(value_monitored='val_accuracy')],
                                 use_multiprocessing=True,
                                 workers=6))
    for fig, value in enumerate(graph):
        plt.figure(fig)
        for index, trained in enumerate(history):
            x = range(1, len(trained.history[value]) + 1)
            plt.plot(x, trained.history[value], label=str(batch_sizes[index]))
        plt.legend()
        plt.title(f"{feature_type} Model {value}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.2)
        plt.savefig(f"results/history_{feature_type}_{value}")
        plt.close(fig)


def train_and_predict(feature_type, batch_size):
    in_train, out_train, in_valid, out_valid, in_test, out_test, encoder = get_data(feature_type)
    labels = encoder.classes_
    num_classes = _check_feature_input(feature_type)
    try:
        model = load_model(f"saved_model/{feature_type}")
    except IOError:
        model = cnn.get_model(feature_type, in_train.shape[1:], num_classes)
        history = model.fit(in_train, out_train,
                            batch_size=batch_size,
                            epochs=100,
                            validation_data=(in_valid, out_valid),
                            callbacks=[cnn.get_early_stop()],
                            use_multiprocessing=True,
                            workers=6)
        data_utils.plot_loss(history, feature_type, save_fig=True)
        model.save(f"saved_model/{feature_type}")
    except Exception as err:
        raise Exception(f"cant get model\n{err}")
    # PREDICTION SECTION
    feature, feature_out, _ = data_utils.file_process_feature(f"7_jackson_2.wav",
                                                              config.FREQUENCY_SAMPLED,
                                                              config.FRAME_MAX_LEN,
                                                              feature_type,
                                                              directory=config.PREDICTIONS_DIR)
    feature = numpy.expand_dims(numpy.expand_dims(feature, axis=-1), axis=0)
    predict = model.predict(feature)
    predict = encoder.inverse_transform(numpy.argmax(predict, axis=1))
    print(f"true: {feature_out}\tprediction: {predict}")

    # Unmark the commands below for run test set and to plot & save confusion matrix
    out_predict = model.predict(in_test, use_multiprocessing=True, workers=6, verbose=1)
    out_predict = numpy.argmax(out_predict, axis=1)
    out_true = numpy.argmax(out_test, axis=1)
    data_utils.plot_confusion_matrix(out_true, out_predict, labels, feature_type, save_fig=True)
    return


if __name__ == "__main__":
    plt = data_utils.get_plt()
    os.makedirs("saved_data", exist_ok=True)
    os.makedirs("saved_model", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # PLOTS FOR MANY BATCH SIZE
    # batch = [2**i for i in range(4, 11)]
    # plot_training_many_batch('mfcc', batch)
    # plot_training_many_batch('mel_spec', batch)

    # TRAIN AND PREDICT WITH THE MOST EFFECTIVE BATCH SIZE
    train_and_predict('mfcc', batch_size=512)
    train_and_predict('mel_spec', batch_size=512)

    # Unmark the command below to plot the confusion matrix for all
    # plt.show()
