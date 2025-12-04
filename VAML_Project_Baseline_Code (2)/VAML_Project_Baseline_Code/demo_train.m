function demo_train()
% Train HOG+SVM and save model + CV table (no detection run).

clc; close all; rng(42,'twister');                       % RNG seed (changeable)

% ---- PATHS ----
posDir = fullfile('data','images','pos');                % pos path (changeable)
negDir = fullfile('data','images','neg');                % neg path (changeable)
outModelDir = fullfile('results','models');              % model out (changeable)
outTableDir = fullfile('results','tables');              % tables out (changeable)
ensure_dir(outModelDir, outTableDir);

% ---- HOG / DATASET ----
ResizeTo     = [64 128];                                 % resize for HOG (changeable)
CellSize     = [8 8];                                    % HOG cell (changeable)
BlockSize    = [2 2];                                    % HOG block in cells (changeable)
BlockOverlap = [1 1];                                    % HOG block overlap (changeable)
NumBins      = 9;                                        % HOG bins (changeable)

fprintf('[1/3] Building dataset ...\n');
[X,y] = build_dataset(posDir, negDir, ...
    'ResizeTo',ResizeTo, 'CellSize',CellSize, ...
    'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap, 'NumBins',NumBins);

% ---- CV ----
C_values  = [0.1 0.3 1 3 10];                            % SVM C grid (changeable)
splitMode = 'holdout';                                   % 'holdout' or 'kfold' (changeable)
kfoldK    = 5;                                           % K for k-fold (changeable)

fprintf('[2/3] Cross-validating ...\n');
if strcmpi(splitMode,'kfold')
    cvRes = crossval_eval(X,y,C_values,'Split','kfold','K',kfoldK,'OutDir',outTableDir,'PrimaryMetric','F1');
else
    cvRes = crossval_eval(X,y,C_values,'Split','holdout','Holdout',0.2,'OutDir',outTableDir,'PrimaryMetric','F1'); % holdout size (changeable)
end
[~,ix]=max(cvRes(:,5)); bestC=cvRes(ix,1);

% ---- TRAIN + SAVE ----
fprintf('[3/3] Training best model (C=%.3f) ...\n', bestC);
model = train_svm(X,y,bestC,'Standardize',true,'ClassNames',[0 1]); % standardize/classes (changeable)
save(fullfile(outModelDir,'model_baseline.mat'),'model');           % model filename (changeable)
fprintf('Saved model to results/models/model_baseline.mat\nDONE.\n');

% ---- helper ----
function ensure_dir(varargin)
for i = 1:nargin
    d = varargin{i};
    if exist(d,'dir'), continue; end
    [status,msg] = mkdir(d); % mkdir creates intermediate folders when needed
    if ~status
        error('Failed to create directory %s: %s', d, msg);
    end
end
end
end
