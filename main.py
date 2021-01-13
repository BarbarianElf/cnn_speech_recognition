"""
main file CNN Speech Recognition project

Created on January 12th 2021

@authors: Niv Ben Ami & Ziv Zango
"""
import data_utils
import config
import cnn

import numpy


if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test, encoder = data_utils.process_feature(config.FREQUENCY_SAMPLED,
                                                                                             config.FRAME_MAX_LEN,
                                                                                             feature_type='mfcc')

    model_mfcc = cnn.get_mfcc_model(x_train.shape[1:], config.NUM_OF_SPEAKERS, learning_rate=0.001)

    history = model_mfcc.fit(x_train, y_train,
                             batch_size=256,
                             epochs=50,
                             validation_data=(x_valid, y_valid),
                             callbacks=[cnn.get_early_stop()])
    data_utils.plot_loss(history)
    y_pred1 = model_mfcc.predict(x_test, use_multiprocessing=True, workers=6, verbose=1)
    y_pred1 = numpy.argmax(y_pred1, axis=1)
    y_true1 = numpy.argmax(y_test, axis=1)
    labels = encoder.classes_
    data_utils.plot_confusion_matrix(y_true1, y_pred1, labels)
    print("hi")
