# Vocal Calculator
**<h2>Table of contents</h2>**

   [Introduction](#Introduction)
   
   [Dataset](#Dataset)
   
   [Code Overview](#Code-Overview)
   
   [Results](#Results)

   [Conclusion](#Conclusion)
   
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
 
 <h3>Sentence segmentation and full-sentence recognition</h3>
 
Instead of supplying individual audio files for each component (digits and operators), an attempt was made to process a single test audio file that contained the whole sentence as a way to improve the project. The goal was to separate the audio file into its component parts before doing recognition on each section.


The librosa.effects.split() method from the Librosa package was used to partition the audio. A threshold of 30 dB served as the foundation for segmentation, which was designed to identify substantial changes in the audio signal. To ensure accurate recognition, each segment's duration was considered, and segments with a minimum duration of 0.2 seconds were removed.
```javascript
import librosa
from scipy.io import wavfile

audio, sr = librosa.load(audio_path, sr=None)    
speech_segments = librosa.effects.split(audio, top_db=30)
MIN_SEGMENT_DURATION = 0.2  # Define the minimum segment duration in seconds

for i, segment in enumerate(speech_segments):
    segment_duration = librosa.get_duration(y=audio[segment[0]:segment[1]], sr=sr)
    if segment_duration >= MIN_SEGMENT_DURATION:
        segment_audio = audio[segment[0]:segment[1]]
        segment_path = f'segment_{i}.wav'
        wavfile.write(segment_path, sr, (segment_audio * 32768).astype(np.int16))

 ```
 
 For additional processing and recognition, each part was stored as a separate audio file. The segment index was indicated by "i" in the file names, which were formatted as "segment_i.wav".

 
 <h2>Results:</h2>
 All the test provided were recognized correctly when we tested on separate audio files for each component (digits and operators).
 Here are some examples of what the output looks like : 
 
 * The files provided contained : 0 * 4
 
![image](https://github.com/ikram28/Vocal-Calculator/assets/86806466/80d58c07-769c-45aa-a117-2e1aa956ffe2)

* The files provided contained : 8 / 5

![image](https://github.com/ikram28/Vocal-Calculator/assets/86806466/87ac5d73-c335-455a-9ecd-82a159ccfaff)

However, when we provided a single audio file and then we segmented it, the recognition results obtained were not satisfactory. The segmentation process introduced additional complexities due to factors such as varying speech speed, overlapping components, and potential misalignment between the segmented components. These factors significantly affected the accuracy of the recognition process.
 Here are some examples of what the output looks like :
* The file we provided contains the following sentence: 4 / 2

![Screenshot_878](https://github.com/ikram28/Vocal-Calculator/assets/86806466/094cc0a6-9ae0-45f1-946f-3f7d80ca9416)

* The file we provided contains the following sentence:  9 + 1

![Screenshot_879](https://github.com/ikram28/Vocal-Calculator/assets/86806466/b41de866-242a-4600-9bd8-5942578b4dc6)

 


<h2>Conclusion:</h2>
In conclusion, this assignment's use of Dynamic Time Warping (DTW) for voice recognition produced successful results when specific audio files for numbers and operators were employed. Based on their MFCC features, the DTW algorithm, in particular the FastDTW optimization, showed its efficacy in precisely matching and identifying spoken numbers and mathematical operators.

The identification outcomes were, however, inadequate when attempting to process a single test audio file including the entire sentence. Additional challenges were created by the segmentation and recognition of whole sentences, including different speech rates, overlapping components, and possible segment alignment issues. These elements have a big impact on how accurate the recognition process was.

The difficulties discovered in segmenting and recognizing whole sentences highlight the need for additional research and the exploration of other methodologies, even though DTW has demonstrated its effectiveness in recognizing individual components. Utilizing complex audio segmentation algorithms and alignment techniques could be one approach to increasing the precision and dependability of entire sentence recognition.






 
