clear all; clc;

%% Arrhythmia Detector

% Load the audio file and convert it to an ECG signal
[file, path] = uigetfile({'*.wav','Audio Files'}, 'Select an audio file');
filePath = fullfile(path, file);
[y, fs] = audioread(filePath);
ecg_signal = y(:,1);

% Step 1: Load the Train and Test CSV files
train_data = csvread('mitbih_train.csv');
test_data = csvread('mitbih_test.csv');

% Step 2: Extract the input signal and class labels from the Train and Test CSV files
X_train = train_data(:, 1:187);
Y_train = categorical(train_data(:, 188));
X_test = test_data(:, 1:187);
Y_test = categorical(test_data(:, 188));

% Step 3: Preprocess the input signals using digital signal processing techniques
% Apply noise filtering using a low-pass filter
[b, a] = butter(4, 0.2, 'low'); % 4th order Butterworth filter with 0.2 cutoff frequency
X_train_filtered = filter(b, a, X_train, [], 2);
X_test_filtered = filter(b, a, X_test, [], 2);

% Normalize the signal to a range of [-1, 1]
X_train_normalized = X_train_filtered ./ max(abs(X_train_filtered), [], 2);
X_test_normalized = X_test_filtered ./ max(abs(X_test_filtered), [], 2);

% Extract features using time-domain and frequency-domain analysis
mean_train = mean(X_train_normalized, 2);
var_train = var(X_train_normalized, [], 2);
mean_test = mean(X_test_normalized, 2);
var_test = var(X_test_normalized, [], 2);

% Prepare the dataset for classification
X_train_features = [mean_train, var_train]; % Features matrix for training set
X_test_features = [mean_test, var_test]; % Features matrix for testing set

% Train the fitcnet() classifier
net = fitcnet(X_train_features, Y_train);

% Test the classifier on the testing set
Y_pred = predict(net, X_test_features);
accuracy = sum(Y_pred == Y_test) / numel(Y_test);

% Step 4: Preprocess the input ECG signal using digital signal processing techniques
% Apply noise filtering using a low-pass filter
[b, a] = butter(4, 0.2, 'low'); % 4th order Butterworth filter with 0.2 cutoff frequency
filtered_signal = filter(b, a, ecg_signal);

% Normalize the signal to a range of [-1, 1]
normalized_signal = filtered_signal ./ max(abs(filtered_signal));

% Step 5: Convert the audio signal to ECG signal file
csvwrite('input_ecg_signal.csv', normalized_signal); % Save the ECG signal as a CSV file
audiowrite('output_ecg_signal.wav', normalized_signal, fs);

% Load the ECG signal file
ecg_data = csvread('input_ecg_signal.csv');

% Extract features from the preprocessed ECG signal (similar to Step 4)
mean_ecg = mean(ecg_data, 2);
var_ecg = var(ecg_data,[],2);
median_ecg = median(ecg_data, 2);
skewness_ecg = skewness(ecg_data, 0, 2);
kurtosis_ecg = kurtosis(ecg_data, 0, 2);

% Prepare the features matrix for classification
features_ecg = [mean_ecg, var_ecg, median_ecg, skewness_ecg, kurtosis_ecg];

% Classify the ECG signal using the trained fitcnet() classifier
label_ecg = predict(net, features_ecg(:, 1:2));
[~, score] = predict(net, X_test_features);
for i = 1:size(X_test_features, 1)
    fprintf('Example %d: Predicted label: %s, Probability: %f\n', i, score(i, 2), score(i, 2));
end
% Convert the value 1 to a categorical value
category_one = categorical(0, [0 1], {'0', '1'});

% Check if the person has arrhythmia or not based on the classification label
if all(label_ecg == category_one)
    fprintf('The ECG signal indicates Arrhythmia\n');
else
    fprintf('The ECG signal does not indicate Arrhythmia. (Normal) \n');
end

% Plot the input audio signal and the preprocessed ECG signal
t_ecg = (0:length(ecg_data)-1)/fs;
figure;
subplot(2, 1, 1);
plot(t_ecg, ecg_signal);
title('Input Audio Signal');
xlabel('Time (s)');
ylabel('Amplitude');
subplot(2, 1, 2);
plot(t_ecg, normalized_signal);
title('Preprocessed ECG Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Display the accuracy of the classifier on the test set
fprintf('Accuracy on test set: %.2f%%\n', accuracy*100);