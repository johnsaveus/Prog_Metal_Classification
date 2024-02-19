## Progressive metal band classification

This is a repository for the semester project of Machine Learning Course in MSc AI NCSR.

The goal is to create a classifier that can categorize progressive metal bands.

The total bands were 5.

### Data collection

Data were collected from youtube URLs. The wav file was extracted from the URL and added to a wav processing software. In the software, a riff was extracted (musical instruments, not voice)

### Feature Extraction

Feautures were extracted using pyAudioAnalysis module [1]. Short term features were extracted and the feature vector consisted of their mean, and std.

[1] https://github.com/tyiannak/pyAudioAnalysis

### Feature Selection

For the feature selection proccess, features were dropped using pearson corellation. For a pair of features that had Pearson > threshold, the first feature was dropped.

### 1st approach: Cross validation on the whole dataset

* Repeated shuffle stratification on the whole dataset (80-20 split).
* Scaling based on training set.
*





