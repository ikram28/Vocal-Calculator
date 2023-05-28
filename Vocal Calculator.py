#!/usr/bin/env python
# coding: utf-8




import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import scipy.io.wavfile
from python_speech_features import mfcc
import speech_recognition as sr
import re
from dtw import dtw


# # Load the dataset and extract the MFCC features:



def load_extractMFCC_save(path):
    my_dict = {}

    for folder in os.listdir(path):
        class_folder = os.path.join(path, folder)
        if os.path.isdir(class_folder):
            my_dict[folder] = []
            for audio_file in os.listdir(class_folder):
                audio_path = os.path.join(class_folder, audio_file)
                freq, audio = scipy.io.wavfile.read(audio_path, mmap=False)
                mfcc_features = mfcc(audio, freq, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=3000, lowfreq=0,
                                     highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False)
                
           
                my_dict[folder].append(mfcc_features)

    return my_dict





my_dict = load_extractMFCC_save(r'C:\Users\ASUS ROG STRIX\Desktop\Projet\RAP\Dataset')







# # Define the test audio file paths for each element in the sentence



number_audio_path = r'C:\Users\ASUS ROG STRIX\Desktop\Projet\RAP\Dataset\Test\zero.wav'
operator_audio_path = r'C:\Users\ASUS ROG STRIX\Desktop\Projet\RAP\Dataset\Test\times.wav'
second_number_audio_path = r'C:\Users\ASUS ROG STRIX\Desktop\Projet\RAP\Dataset\Test\four.wav'



# # Extract MFCC features for the test audio files



freq1, number_audio = scipy.io.wavfile.read(number_audio_path, mmap=False)
freq2, operator_audio = scipy.io.wavfile.read(operator_audio_path, mmap=False)
freq3, second_number_audio = scipy.io.wavfile.read(second_number_audio_path, mmap=False)

number_mfcc_features = mfcc(number_audio,freq1, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=3000, lowfreq=0,
                            highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False)
operator_mfcc_features = mfcc(operator_audio, freq2, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=3000, lowfreq=0,
                              highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False)
second_number_mfcc_features = mfcc(second_number_audio,freq3, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=3000, lowfreq=0,
                                   highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False)



#  Perform recognition for the number



best_distance_number = float('inf')
best_number = None
for class_name, class_mfccs in my_dict.items():
    if class_name in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
        for class_mfcc in class_mfccs:
            distance, _ = fastdtw(number_mfcc_features, class_mfcc, dist=euclidean)
            if distance < best_distance_number:
                best_distance_number = distance
                best_number = class_name


#  Perform recognition for the operator


best_distance_operator = float('inf')
best_operator = None
for class_name, class_mfccs in my_dict.items():
    if class_name in ['plus', 'minus', 'times', 'dividedBy']:
        for class_mfcc in class_mfccs:
            distance, _ = fastdtw(operator_mfcc_features, class_mfcc, dist=euclidean)
            if distance < best_distance_operator:
                best_distance_operator = distance
                best_operator = class_name


#  Perform recognition for the second number



best_distance_second_number = float('inf')
best_second_number = None
for class_name, class_mfccs in my_dict.items():
    if class_name in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
        for class_mfcc in class_mfccs:
            distance, _ = fastdtw(second_number_mfcc_features, class_mfcc, dist=euclidean)
            if distance < best_distance_second_number:
                best_distance_second_number = distance
                best_second_number = class_name


# # Number and operator mapping



number_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
number1 = number_mapping.get(best_number)
number2 = number_mapping.get(best_second_number)

operator_mapping = {'plus': '+', 'minus': '-', 'times': '*', 'dividedBy': '/'}
operator = operator_mapping.get(best_operator)


# Perform the math operation and Print the recognized number, operator, and result



if number1 is not None and operator is not None and number2 is not None:
    if operator == '+':
        result = number1 + number2
    elif operator == '-':
        result = number1 - number2
    elif operator == '*':
        result = number1 * number2
    elif operator == '/':
        result = number1 / number2

    print("Recognized Number: ", number1)
    print("Recognized Operator: ", operator)
    print("Recognized Second Number: ", number2)
    print("Result: ", result)

