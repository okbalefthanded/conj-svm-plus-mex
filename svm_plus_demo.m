% Okba BEKHELIFI <okba.bekhelifi@univ-usto.dz>
% 11-20-2017
%% Train & Test data (Toy Data)
labels = [ones(10,1);-1*ones(10,1)];
n_features = 100;
n_instances  = 20;
data = rand(n_instances, n_features);

x = data([1:5,10:15],:);
label_train = labels([1:5,10:15]);
x_star = rand(length(label_train), n_features);
x_test = data([6:9,16:20],:);
label_test = labels([6:9,16:20]);
%% Train and predict SVM
tic;
model_svm = svmtrain(label_train, x);
[predicted_label, accuracy, decision_values] = svmpredict(label_test, x_test, model_svm);
toc
%% Train and predict SVM+ (aSMO)
tic
model_svm_plus = svm_train_plus(label_train, x, x_star, '-s 5 -t 0 -a -1 -T 0 -c 0.1 -C 0.01');
[predicted_label_plus, accuracy_plus, decision_values_plus] = svm_predict_plus(label_test, x_test, model_svm_plus);
toc
%% Train and predict SVM+ (caSMO)
tic
model_svm_plus = svm_train_plus(label_train, x, x_star, '-s 5 -t 0 -a 1 -T 0 -c 0.1 -C 0.01');
[predicted_label_plus, accuracy_plus, decision_values_plus] = svm_predict_plus(label_test, x_test, model_svm_plus);
toc