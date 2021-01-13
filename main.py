"""
main file CNN Speech Recognition project

Created on January 12th 2021

@authors: Niv Ben Ami & Ziv Zango
"""
import data_utils
import config
import cnn
import os
import numpy


def get_data(feature_type=None):
    try:
        x_data, y_data = numpy.load("x_data.npy"), numpy.load("y_data.npy")
        x_test, y_test = numpy.load("x_test.npy"), numpy.load("y_test.npy")
    except IOError:
        x_data, x_test, y_data, y_test = data_utils.process_feature(config.FREQUENCY_SAMPLED,
                                                                    config.FRAME_MAX_LEN,
                                                                    feature_type=feature_type,
                                                                    num_for_test=config.NUM_OF_TEST)
        x_data, x_test = data_utils.expand_dim(x_data, x_test)
    y_data, y_test, _encoder = data_utils.data_encoder_and_categorized(y_data, y_test)
    x_train, x_valid, y_train, y_valid = data_utils.train_valid_split(x_data, y_data, percent=config.VALID_PERCENT)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, _encoder


if __name__ == "__main__":
    plt = data_utils.get_plt()
    os.makedirs("results", exist_ok=True)

    in_train, out_train, in_valid, out_valid, in_test, out_test, encoder = get_data('mfcc')
    model = cnn.get_mfcc_model(in_train.shape[1:], config.NUM_OF_SPEAKERS, learning_rate=0.001)
    history = model.fit(in_train, out_train,
                        batch_size=800,
                        epochs=100,
                        validation_data=(in_valid, out_valid),
                        callbacks=[cnn.get_early_stop()],
                        use_multiprocessing=True,
                        workers=6,)

    out_predict = model.predict(in_test, use_multiprocessing=True, workers=6, verbose=1)
    out_predict = numpy.argmax(out_predict, axis=1)
    out_true = numpy.argmax(out_test, axis=1)
    labels = encoder.classes_

    data_utils.plot_loss(history)
    data_utils.plot_confusion_matrix(out_true, out_predict, labels)

    # model = cnn.get_mel_spec_model(in_train.shape[1:], config.NUM_OF_DIGITS, learning_rate=0.001)
    print("hi")
