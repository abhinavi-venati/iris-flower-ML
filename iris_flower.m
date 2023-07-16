load fisheriris
X = meas;
y = species;

X_mean = mean(X);
X_std = std(X);
X = (X - X_mean) ./ X_std;

k = 5;
model = fitcknn(X, y, 'NumNeighbors', k);

cv = cvpartition(size(X, 1), 'KFold', 5);
accuracy = zeros(cv.NumTestSets, 1);
confusion_matrix = zeros(3, 3);

for i = 1:cv.NumTestSets
    X_train = X(training(cv, i), :);
    y_train = y(training(cv, i), :);
    X_test = X(test(cv, i), :);
    y_test = y(test(cv, i), :);

    model = fitcknn(X_train, y_train, 'NumNeighbors', k);

    y_pred = predict(model, X_test);
    accuracy(i) = sum(strcmp(y_pred, y_test)) / numel(y_test);
    
    C = confusionmat(y_test, y_pred, 'Order', {'setosa', 'versicolor', 'virginica'});
    confusion_matrix = confusion_matrix + C;
end

mean_accuracy = mean(accuracy);
disp(['Mean Accuracy: ' num2str(mean_accuracy)]);

disp('Confusion Matrix:');
disp(confusion_matrix);

class_labels = {'setosa', 'versicolor', 'virginica'};
precision = zeros(3, 1);
recall = zeros(3, 1);
F1_score = zeros(3, 1);

for i = 1:3
    TP = confusion_matrix(i, i);
    FP = sum(confusion_matrix(:, i)) - TP;
    FN = sum(confusion_matrix(i, :)) - TP;
    
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    F1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    
    disp(['Class: ' class_labels{i}]);
    disp(['Precision: ' num2str(precision(i))]);
    disp(['Recall: ' num2str(recall(i))]);
    disp(['F1 Score: ' num2str(F1_score(i))]);
end

save('iris_model.mat', 'model', 'X_mean', 'X_std');

disp('Trained model saved successfully.');

figure;
gscatter(X(:, 1), X(:, 2), y, 'rgb', 'o', 7);
xlabel('Sepal Length (cm)');
ylabel('Sepal Width (cm)');
title('Iris Flower Dataset');

load('iris_model.mat', 'model', 'X_mean', 'X_std');

predictSpecies(X_mean, X_std, model);

function predictSpecies(X_mean, X_std, model)
    prompt = {'Enter sepal length (cm):', 'Enter sepal width (cm):', 'Enter petal length (cm):', 'Enter petal width (cm):'};
    title = 'Iris Flower Classification';
    dims = [1 50];
    userInput = inputdlg(prompt, title, dims);
    userMeasurements = str2double(userInput)';
    userMeasurements = (userMeasurements - X_mean) ./ X_std;
    predictedSpecies = predict(model, userMeasurements);
    disp(['Predicted Species: ' char(predictedSpecies)]);
end
