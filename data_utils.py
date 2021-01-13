import os
import librosa
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.utils import to_categorical

from mfcc_utils import mf_cepstral_coefficients as mfcc
from mfcc_utils import mel_spectrogram
from mfcc_utils import power_to_db
import config


def plot_confusion_matrix(y_true, y_pred, labels, color_map='viridis'):
    """
    plots the confusion matrix
    """
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(matrix, cmap=color_map)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, true_label in enumerate(matrix):
        for j, predicted_label in enumerate(true_label):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")


def plot_loss(trained):
    plt.figure(figsize=(10, 7))
    plt.plot(trained.history['loss'], c="darkblue")
    plt.plot(trained.history['val_loss'], c="crimson")
    plt.legend(["Train", "Validation"])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.2)


def _get_wav_files(dir_path):
    # insert to list all the files with wav extension
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".wav"):
            files.append(file)
    return files


def process_feature(fs, max_len, feature_type=None):
    x_data, x_test, y_data, y_test = [], [], [], []
    for file in _get_wav_files(config.RECORDING_DIR):
        file_name = file.split('.')[0].split('_')
        word, speaker, index = file_name[0], file_name[1], file_name[2]
        sound_data, _ = librosa.core.load(config.RECORDING_DIR + file, sr=fs)
        if feature_type == "mfcc":
            filter_num = config.MFCC_FILTER_NUM
            feature = mfcc(sound_data, fs, pre_emphasis=False, dct_filters_num=filter_num, normalized=True)
            feature_out = speaker
        elif feature_type == "mel_spec":
            filter_num = config.MEL_FITER_NUM
            spectrogram, _ = mel_spectrogram(sound_data, fs, mel_filters=filter_num, normalized=True)
            feature = power_to_db(spectrogram)
            feature_out = word
        else:
            raise ValueError('feature_type must be `mfcc` or `mel_spec`')
        if feature.shape[1] < max_len:
            feature = numpy.pad(feature, ((0, 0), (0, max_len - feature.shape[1])))
        else:
            feature = feature[:, :max_len]
        if int(index) < 5:
            x_test.append(feature)
            y_test.append(feature_out)
        else:
            x_data.append(feature)
            y_data.append(feature_out)

    x_data, x_test = numpy.expand_dims(numpy.array(x_data), axis=-1), numpy.expand_dims(numpy.array(x_test), axis=-1)

    encoder = LabelEncoder()
    encoder.fit(numpy.array(y_data))

    y_data = to_categorical(encoder.transform(numpy.array(y_data)))
    y_test = to_categorical(encoder.transform(numpy.array(y_test)))

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data,
                                                          test_size=1 / 9, random_state=True, shuffle=True)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, encoder