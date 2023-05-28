# Vocal Calculator
**<h2>Table of contents</h2>**

   [Introduction:](#Introduction)
   
   [Dataset:](#Dataset)
   
   [Code Overview:](#Code-Overview)
   
   [Results:](#Results)

   [Conclusion:](#Conclusion)
   
<h2>Introduction: </h2>
The goal of this assignment is to create a speech recognition system that uses audio input to perform mathematical operations. Recognizing spoken numbers and operators is the goal, followed by computing the associated math operation.

<h2>Dataset:</h2>
The dataset used for the speech recognition assignment consists of 14 folders, each representing digits (0-9) and mathematical operators (/, -, *,+ ).
There are five audio files each folder, for a total of 70 audio files. The dataset is structured, with distinct folders for each class, making it simple to retrieve and analyze the audio samples.

<h2>Code Overview:</h2>
- The first thing we do is loading the audios contained in the dataset, then we extract their corresponding MFFC features and save them in a dictionary.

 ```javascript
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
 ```
 
 
 - Next, we define the test audio file paths for each element in the sentence: 
    - number_audio_path : the path for the first digit of the operation
    - operator_audio_path : the path for the operators (+, - , *, /)
    - second_number_audio_path : the path for the second digit of the operation
 - After that, we extract MFCC features for those three files then we perform the recognition using the DTW algorithm (Dynamic Time Warping) which is a technique used for comparing two temporal sequences, such as audio signals or time series data, by aligning them to find the optimal alignment that minimizes the distance measure. It allows for flexible matching of sequences with different lengths or temporal variations. In this case, DTW is employed to find the best match between the test audio's operator MFCC features and the training samples' MFCC features, enabling accurate operator recognition.
In this assignment, we have used the FastDTW which is an optimization technique for DTW that significantly reduces the computational complexity of the algorithm while providing a near-optimal alignment. 
The following code snippet demonstrates the implementation for recognizing the operator: 

 ```javascript
best_distance_operator = float('inf')
best_operator = None
for class_name, class_mfccs in my_dict.items():
    if class_name in ['plus', 'minus', 'times', 'dividedBy']:
        for class_mfcc in class_mfccs:
            distance, _ = fastdtw(operator_mfcc_features, class_mfcc, dist=euclidean)
            if distance < best_distance_operator:
                best_distance_operator = distance
                best_operator = class_name
 ```
 
 The above code loops through the training set of MFCC features kept in the my_dict dictionary. It concentrates particularly on the classes that represent the operators (such as "plus," "minus," "times," and "dividedBy"). It determines the DTW distance for each class between each training sample's MFCC features and the operator_mfcc_features (extracted from the test audio). We use the Euclidean distance as the distance measure.
- After selecting the training data that best matches the test audio files based on the DTW distance, we map the recognized numbers and operators using this predefined dictionaries :

```javascript
number_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
operator_mapping = {'plus': '+', 'minus': '-', 'times': '*', 'dividedBy': '/'}
 ```
 
 - The final step is performing the math operation based on the recognized numbers and operator and then returning the recognized number, operator, and the computed result.
 
 <h2>Results:</h2>
 All the test provided were recognized correctly.
 Here are some examples of what the output looks like :
 
![image](https://github.com/ikram28/Vocal-Calculator/assets/86806466/80d58c07-769c-45aa-a117-2e1aa956ffe2)
![image](https://github.com/ikram28/Vocal-Calculator/assets/86806466/87ac5d73-c335-455a-9ecd-82a159ccfaff)

<h2>Conclusion:</h2>
In conclusion, this assignment's use of DTW for voice recognition has produced encouraging outcomes. Based on its MFCC properties, the DTW algorithm, more specifically the FastDTW optimization, has proven its capability to effectively match and detect spoken numbers and mathematical operators.


Even while DTW produces positive results, the recognition accuracy could still use some improvement. Investigating additional methods and algorithms frequently utilized in voice recognition tasks is one option that could be taken. For instance, utilizing deep learning models' capacity to recognize intricate connections and patterns in sequential data, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs), may enhance recognition performance.

As an improvement to this project, the system may be given a single test file containing the complete sentence and segment it into its component parts (digits and operators) before executing the recognition procedure.
This approach would more closely resemble real-world scenarios where continuous speech is encountered, and it would require additional steps such as speech segmentation and alignment to accurately identify and recognize the individual components of the sentence.



 
