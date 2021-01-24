"""
config file
for CNN Speech Recognition project

Created on January 12th 2021

@authors: Niv Ben Ami & Ziv Zango
"""
# recording directory path
RECORDING_DIR = "./recordings/"
PREDICTIONS_DIR = "./predictions/"
# frequency sampled for audio
FREQUENCY_SAMPLED = 8000

# properties of mfcc/mel-spectrogram
FRAME_MAX_LEN = 85
MFCC_FILTER_NUM = 20
MEL_FITER_NUM = 40

# number of recording for each word
NUM_OF_DATA = 50
# TRAIN 80% VALID 10% TEST 10%
NUM_OF_TEST = NUM_OF_DATA * 0.1
NUM_OF_VALID = NUM_OF_DATA * 0.1
VALID_PERCENT = NUM_OF_VALID / (NUM_OF_DATA - NUM_OF_TEST)

# speaker and word number for classification
NUM_OF_SPEAKERS = 40
NUM_OF_WORDS = 10

