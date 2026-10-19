close all; clear; clc;

%% -------- CONFIGURATION --------
% Define paths
dataset_zip = "C:\Users\HP\OneDrive\Desktop\bigdataset_158.zip";
unzip_folder = "C:\Users\HP\OneDrive\Desktop\bigdataset_158";

% Unzip dataset if not already unzipped
if ~exist(unzip_folder, 'dir')
    unzip(dataset_zip, unzip_folder);
    disp('Dataset successfully unzipped!');
else
    disp('Dataset was already unzipped.');
end

% Image processing parameters
img_height = 128; img_width = 128;

% PCA and LDA dimensions
k_pca_main = 80;
k_pca_lda  = 30;

% Number of nearest neighbors for classification
k = 5;

% Training/test split
train_ratio = 0.8;

% Number of samples to load from each class
num_male = 500; num_female = 500;

% Dataset paths
male_path = fullfile(unzip_folder, 'bigdataset_158', 'male');
female_path = fullfile(unzip_folder, 'bigdataset_158', 'female');

%% -------- LOAD IMAGES --------
data = []; labels = [];

% Load Male (Label = 0)
male_imgs = dir(fullfile(male_path, '*.jpg'));
male_imgs = male_imgs(1:min(num_male, length(male_imgs)));

for i = 1:length(male_imgs)
    img = imread(fullfile(male_path, male_imgs(i).name));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, [img_height img_width]);
    img = adapthisteq(img);
    vec = double(reshape(img, [], 1));
    data = [data vec];
    labels = [labels; 0];
end

% Load Female (Label = 1)
female_imgs = dir(fullfile(female_path, '*.jpg'));
female_imgs = female_imgs(1:min(num_female, length(female_imgs)));

for i = 1:length(female_imgs)
    img = imread(fullfile(female_path, female_imgs(i).name));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, [img_height img_width]);
    img = adapthisteq(img);
    vec = double(reshape(img, [], 1));
    data = [data vec];
    labels = [labels; 1];
end

X = data;
Y = labels';
n = size(X, 2);
fprintf('Loaded %d images: %d Male, %d Female\n', n, sum(Y==0), sum(Y==1));

%% -------- PCA --------
mean_face = mean(X, 2);
Xc = X - mean_face;
S = (Xc' * Xc) / (n - 1);
[Veig, D] = eig(S);
[~, order] = sort(diag(D), 'descend');
Veig = Veig(:, order);
U = Xc * Veig;

for i = 1:size(U,2)
    U(:,i) = U(:,i) / norm(U(:,i));
end

Upca_main = U(:, 1:min(k_pca_main, size(U, 2)));
Xpca_main = Upca_main' * Xc;

Upca_lda = U(:, 1:min(k_pca_lda, size(U, 2)));
Xpca_lda = Upca_lda' * Xc;

%% -------- Train/Test Split --------
rng(42);
perm = randperm(n);
split = floor(train_ratio * n);
train_idx = perm(1:split);
test_idx  = perm(split+1:end);

X_train_pca = Xpca_main(:, train_idx);
X_test_pca  = Xpca_main(:, test_idx);
X_train_lda_input = Xpca_lda(:, train_idx);
X_test_lda_input  = Xpca_lda(:, test_idx);
Y_train = Y(train_idx);
Y_test  = Y(test_idx);

%% -------- LDA --------
classes = unique(Y_train);
mean_total = mean(X_train_lda_input, 2);
Sw = zeros(k_pca_lda); Sb = zeros(k_pca_lda);

for i = 1:length(classes)
    idx = find(Y_train == classes(i));
    Xi = X_train_lda_input(:, idx);
    mi = mean(Xi, 2);
    ni = length(idx);
    Sw = Sw + (Xi - mi) * (Xi - mi)';
    Sb = Sb + ni * (mi - mean_total) * (mi - mean_total)';
end

[Wlda_temp, D] = eig(Sb, Sw);
[~, order] = sort(diag(D), 'descend');
W_lda = Wlda_temp(:, order(1));

X_train_lda = W_lda' * X_train_lda_input;
X_test_lda  = W_lda' * X_test_lda_input;

%% -------- k-NN (Cosine) --------
function predicted = knn_cosine(X_train, Y_train, X_test, k)
    num_test = size(X_test, 2);
    predicted = zeros(1, num_test);
    for i = 1:num_test
        x = X_test(:, i) / norm(X_test(:, i));
        xt = X_train ./ vecnorm(X_train);
        cos_dists = 1 - x' * xt;
        [~, idx] = mink(cos_dists, k);
        predicted(i) = mode(Y_train(idx));
    end
end

Y_pred_pca = knn_cosine(X_train_pca, Y_train, X_test_pca, k);
Y_pred_lda = knn_cosine(X_train_lda, Y_train, X_test_lda, k);

%% -------- Evaluation --------
acc_pca = sum(Y_pred_pca == Y_test) / length(Y_test) * 100;
acc_lda = sum(Y_pred_lda == Y_test) / length(Y_test) * 100;

fprintf('\nGENDER CLASSIFICATION RESULTS (Split: %.0f%% train / %.0f%% test):\n', train_ratio*100, 100-train_ratio*100);
fprintf('PCA Accuracy: %.2f%%\n', acc_pca);
fprintf('LDA Accuracy: %.2f%%\n', acc_lda);

fprintf('\nConfusion Matrix (PCA):\n');
disp(confusionmat(Y_test, Y_pred_pca));

fprintf('Confusion Matrix (LDA):\n');
disp(confusionmat(Y_test, Y_pred_lda));

%% -------- ROC Curve and FPR/TPR --------
% Convert to column vector for perfcurve
Y_test_col = Y_test(:);
Y_pred_pca_col = Y_pred_pca(:);
Y_pred_lda_col = Y_pred_lda(:);

% Calculate ROC for PCA
[fp_pca, tp_pca, ~, auc_pca] = perfcurve(Y_test_col, Y_pred_pca_col, 1);

% Calculate ROC for LDA
[fp_lda, tp_lda, ~, auc_lda] = perfcurve(Y_test_col, Y_pred_lda_col, 1);

% Plot ROC curves
figure;
plot(fp_pca, tp_pca, 'b-', 'LineWidth', 2); hold on;
plot(fp_lda, tp_lda, 'r--', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves (1000 Image Dataset)');
legend(sprintf('PCA (AUC = %.2f)', auc_pca), sprintf('LDA (AUC = %.2f)', auc_lda));
grid on;
