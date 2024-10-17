## Abstract
This report describes developing and evaluating two neural network architectures designed to classify music genres in the GTZAN dataset, which consists of ten au- dio types. The first architecture is a Recurrent Neural Network (RNN) utilising Long Short-Term Memory (LSTM) units. The second model builds on the first model by incorporating a Generative Adversarial Network (GAN) to augment the training dataset by synthesising new audio samples, thus effectively scaling up the size of the dataset. According to the results, using the GAN to increase the training dataset leads to a slight increase in the test accuracy of the model.
## 1. Introduction
In this report, two RNN classifiers utilising LSTM will be built for the GTZAN music dataset from Kaggle. The original and GAN-enhanced training sets will be used separately to compare the performance differences.
## 2. Methodologies
### 2.1. Mel-Spectrograms
The first step of this report is to transfer the audio files to Mel-Spectrograms. Mel spectrograms are favoured for their detailed time-frequency representation of sound, which closely reflects human auditory perception. This visualisation is well suited for complex audio tasks such as music classification, as it improves the accuracy of deep learning models in feature recognition and classifica- tion. In contrast, Mel spectrograms are more effective than MFCC in these applications [1]. The example of the Mel-Spectrogram in this dataset is shown in the appendix (Appendix A).
### 2.2. RNN and LSTMs
Recurrent Neural Networks (RNNs) are a class of neural networks that are pivotal in modelling sequence data. Unlike traditional neural networks, which assume that all inputs (and outputs) are independent of each oth- er, RNNs operate under the assumption that the inputs are related to each other in a sequence. Long Short-Term Memory Networks (LSTMs) are special RNNs designed to remember information for long periods [2].
### 2.3. Rectified Linear Unit
This report will apply the Rectified Linear Unit (ReLU) to improve the models’ performance. ReLU models learn faster and perform better on various tasks than models using sigmoid or tanh. This is due to the alleviation of the vanishing gradient problem and the computational efficiency of ReLU [3].
### 2.4. Batch Normalisation
Batch normalisation has also been applied in the models. Batch normalisation is a transformative technique in neural network training designed to accelerate cover glance, improve the stability of the learning process, and lower the use of high- er learning rates [4].
### 2.5. GAN
GAN can generate new data samples that can be used to train other machine learning models. This is especially useful when additional training data is needed but difficult to collect [5]. The training set in this report includes 699 audio data. As suggested in the coursework introduction, this report will gen- erate 700 pieces of data (each category 70) to augment the training data. Also, this report will try to generate 1500 new data to compare the performance.
## 3. Results
### 3.1. RNN with Original Training Set
In the first model, the test accuracy stands at 52.48%. The confusion matrix (Appendix B) shows a higher rate of correct predictions in Classical and metal genres. However, this model struggled with disco and rock genres, failing to identify tracks in these genres correctly.
### 3.2. RNN with Gan
The second model achieved a test set accuracy of 58.42%. The confusion matrix of this model (Appendix C) shows that the model correctly predicts truer classical and metal music but also makes mistakes identifying some other labels to these two labels. However, this model weakly classifies jazz with no correct identification. The accuracy of the model when gener- ating 1500 is 51.49%, which shows the increase in generated data didn’t equal the increase in model performance.
## 4. Conclusion
The result shows the training set adjusted by Gan can achieve higher accuracy on the test set. However, these mod- els all show worse ability than the CNN in the previous report. They also show the limitations of the Gan dataset. Sugges- tions for improvement include using a more extensive dataset, or CNN should be more suitable for this task and dataset.


## 5. Reference
[1] K. Doshi, “Audio deep learning made simple (part 2): Why Mel Spectrograms perform better,” Medium, https://towardsdatascience.com/audio-deep-learning- made-simple-part-2-why-mel-spectrograms-perform- better-aad889a93505 (accessed Apr. 12, 2024).

[2] C. Olah, “Understanding LSTM networks,” Under- standing LSTM Networks -- colah’s blog, https://colah.github.io/posts/2015-08-Understanding- LSTMs/ (accessed Apr. 21, 2024).

[3] J. Brownlee, “A gentle introduction to the rectified linear unit (ReLU),” MachineLearningMastery.com, https://machinelearningmastery.com/rectified-linear- activation-function-for-deep-learning-neural-networks/ (accessed Apr. 12, 2024).

[4] M. Chablani, “Batch normalisation,” Medium, https://towardsdatascience.com/batch-normalization- 8a2e585775c9 (accessed Apr. 21, 2024).

[5] K. Ahirwar, Generative Adversarial Networks Pro- jects: Build Next-Generation Generative Models Using Tensorflow and Keras. Birmingham, UK: Packt Publish- ing, 2019.

## Appendix

#### Appendix A:
![image](https://github.com/mengchelee/Advanced-Music-Genre-Classification/blob/main/appendixa.png)
#### Appendix B:
![image](https://github.com/mengchelee/Advanced-Music-Genre-Classification/blob/main/appendixb.png)
#### Appendix C:
![image](https://github.com/mengchelee/Advanced-Music-Genre-Classification/blob/main/appendixc.png)

