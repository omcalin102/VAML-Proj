function make_figures(varargin)
% Create HOG visual, CV bar chart over C values, and runtime plots.
%
% OUTPUTS (in <root>/results/figs/<RunTag>/):
%   - hog_gradient.png            (HOG block visualization from a pedestrian window)
%   - cv_f1_by_C.png              (F1 vs C bar chart with value labels)
%   - cv_runtime_by_C.png         (train + predict time per C)
%   - cv_results_by_C.csv         (per-C metrics & timings)
%
% USAGE:
%   addpath(genpath('/MATLAB Drive/VAML_Project_Baseline_Code'));
%   make_figures('RunTag','run_figs', 'Root', '/MATLAB Drive/VAML_Project_Baseline_Code');
%
% NOTES:
%   - Uses existing dataset builder if available (build_dataset.m).
%   - If you don't have build_dataset, it falls back to extracting HOG
%     on-the-fly from pos/neg image folders: data/images/pos, data/images/neg.
%   - Uses MATLAB's extractHOGFeatures for the HOG visualization.
%
% KEY DIALS YOU CAN CHANGE QUICKLY:
%   Cs:        vector of C values for CV (default [0.1 0.3 1 3 10])
%   K:         K-folds for CV (default 5)
%   BaseWindow, CellSize: HOG geometry (must match the rest of your pipeline)

p = inputParser;
addParameter(p,'Root','/MATLAB Drive/VAML_Project_Baseline_Code');     % <-- change if needed
addParameter(p,'RunTag','run_figs');                                   % outputs go to results/figs/<RunTag>
addParameter(p,'Cs',[0.1 0.3 1 3 10]);                                 % C grid
addParameter(p,'K',5);                                                 % K-folds
addParameter(p,'BaseWindow',[64 128]);                                 % HOG window
addParameter(p,'CellSize',[8 8]);                                      % HOG cell size
parse(p,varargin{:}); args = p.Results;

root   = args.Root;
outDir = fullfile(root, 'results', 'figs', args.RunTag);
if ~exist(outDir,'dir'), mkdir(outDir); end

% ---------------------------------------------------------------------
% 1) HOG GRADIENT VISUALIZATION (from a typical 64×128 pedestrian window)
% ---------------------------------------------------------------------
sampleImgPath = find_sample_pedestrian(root);
I = imread(sampleImgPath);
I = im2uint8(I);
% resize to BaseWindow keeping aspect: crop center to match ratio
Iwin = prepare_window(I, args.BaseWindow);
[hogFeature, hogVis] = extractHOGFeatures(Iwin, 'CellSize', args.CellSize);

fh1 = figure('Visible','off');
imshow(Iwin); hold on;
plot(hogVis); title(sprintf('HOG visualization (%dx%d, cell %dx%d)', ...
    args.BaseWindow(1), args.BaseWindow(2), args.CellSize(1), args.CellSize(2)));
exportgraphics(fh1, fullfile(outDir,'hog_gradient.png'), 'Resolution', 180);
close(fh1);

% ---------------------------------------------------------------------
% 2) CROSS-VALIDATION ACROSS C VALUES (F1 per C) + RUNTIME PLOTS
% ---------------------------------------------------------------------
% Try to use existing builder if available
posDir = fullfile(root, 'data', 'images', 'pos');
negDir = fullfile(root, 'data', 'images', 'neg');
use_builder = exist('build_dataset','file')==2;

if use_builder
    fprintf('[CV] Using build_dataset(...) from %s\n', which('build_dataset'));
    [X,y] = build_dataset(posDir, negDir);  %#ok<ASGLU>
    % If your build_dataset already returns HOG, we'll assume it matches BaseWindow/CellSize.
    % Otherwise, you can switch to the fallback path below by forcing use_builder=false.
else
    fprintf('[CV] build_dataset.m not found; extracting HOG on the fly...\n');
    [X,y] = build_hog_from_folders(posDir, negDir, args.BaseWindow, args.CellSize);
end

% Some datasets can be imbalanced; stratify folds
rng(1);
cv = cvpartition(y, 'KFold', args.K, 'Stratify', true);

Cs = args.Cs(:)';
results = struct('C',[],'Acc',[],'Prec',[],'Rec',[],'F1',[], ...
                 'TrainSec',[],'PredSec',[],'MsPerSample',[]);
rows = {};

for i = 1:numel(Cs)
    C = Cs(i);
    tTrain = 0; tPred = 0;
    TP=0; FP=0; FN=0; N=0;

    for k = 1:cv.NumTestSets
        tr = training(cv,k);
        te = test(cv,k);

        Xt = X(tr,:); yt = y(tr);
        Xv = X(te,:); yv = y(te);

        tic;
        mdl = fitcsvm(Xt, yt, 'KernelFunction','linear', 'BoxConstraint', C, ...
                      'Standardize', true, 'ClassNames', unique(y));
        tTrain = tTrain + toc;

        tic;
        yhat = predict(mdl, Xv);
        tPred = tPred + toc;

        % confusion stats
        [tp,fp,fn] = binary_counts(yv, yhat);
        TP=TP+tp; FP=FP+fp; FN=FN+fn; N = N + numel(yv);
    end

    Acc  = (TP+ (N-TP-FP-FN))/N;
    Prec = TP/(TP+FP+eps);
    Rec  = TP/(TP+FN+eps);
    F1   = 2*Prec*Rec/(Prec+Rec+eps);
    MsPerSample = 1000 * (tPred / N);

    results(i).C = C;
    results(i).Acc = Acc; results(i).Prec = Prec; results(i).Rec = Rec; results(i).F1 = F1;
    results(i).TrainSec = tTrain; results(i).PredSec = tPred; results(i).MsPerSample = MsPerSample;

    rows(end+1,:) = {C, Acc, Prec, Rec, F1, tTrain, tPred, MsPerSample}; %#ok<AGROW>
    fprintf('C=%.3f | Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f | Train=%.2fs Pred=%.2fs (%.2f ms/sample)\n', ...
        C, Acc, Prec, Rec, F1, tTrain, tPred, MsPerSample);
end

% Save CSV
T = cell2table(rows, 'VariableNames', ...
    {'C','Accuracy','Precision','Recall','F1','TrainSeconds','PredictSeconds','MsPerSample'});
writetable(T, fullfile(outDir,'cv_results_by_C.csv'));

% ---- Figure: F1 vs C (bar) ----
fh2 = figure('Visible','off');
F1s = [results.F1];
bar(C2str(Cs), F1s, 'EdgeColor',[0 0 0]);
ylabel('F1 score');
xlabel('C (BoxConstraint)');
title(sprintf('Cross-validation F1 across C (K=%d)', args.K));
ylim([0, 1]); grid on; box on;
text(1:numel(Cs), F1s + 0.02, compose('%.3f',F1s), 'HorizontalAlignment','center', 'FontSize',10);
exportgraphics(fh2, fullfile(outDir,'cv_f1_by_C.png'), 'Resolution', 180);
close(fh2);

% ---- Figure: runtime per C (stacked bars: Train + Predict) ----
fh3 = figure('Visible','off');
Train = [results.TrainSec];
Pred  = [results.PredSec];
b = bar(C2str(Cs), [Train(:) Pred(:)], 'stacked');
ylabel('Time (seconds)');
xlabel('C (BoxConstraint)');
title(sprintf('Runtime per C (K=%d folds)', args.K));
legend({'Train (total)','Predict (total)'}, 'Location','northwest');
grid on; box on;
exportgraphics(fh3, fullfile(outDir,'cv_runtime_by_C.png'), 'Resolution', 180);
close(fh3);

fprintf('Saved all figures & CSV to: %s\n', outDir);
end

% ==================== helpers ====================

function sample = find_sample_pedestrian(root)
% Finds a sample pedestrian image for HOG visual.
candidates = [ ...
    dir(fullfile(root,'pedestrian','pedestrian','*.jpg')); ...
    dir(fullfile(root,'pedestrian','pedestrian','*.png')); ...
    dir(fullfile(root,'data','images','pos','*.jpg')); ...
    dir(fullfile(root,'data','images','pos','*.png'))];
assert(~isempty(candidates), 'No sample pedestrian images found.');
% pick a mid-range image for reproducibility
sample = fullfile(candidates( min(5, numel(candidates)) ).folder, ...
                  candidates( min(5, numel(candidates)) ).name);
end

function Iwin = prepare_window(I, baseWin)
% Resize/crop image to exactly [W H] base window (keep person roughly centered).
W = baseWin(1); H = baseWin(2);
% If image is too small, pad; else center-crop then resize
[h,w,~] = size(I);
if h < H || w < W
    I = imresize(I, [max(h,H) max(w,W)]);
end
% center-crop to aspect ratio of baseWin
targetAR = W/H;
curAR = w/h;
if curAR > targetAR
    % too wide -> crop width
    newW = round(h * targetAR);
    x1 = round((w - newW)/2) + 1;
    I = I(:, x1:x1+newW-1, :);
else
    % too tall -> crop height
    newH = round(w / targetAR);
    y1 = round((h - newH)/2) + 1;
    I = I(y1:y1+newH-1, :, :);
end
Iwin = imresize(I, [H W]);
end

function [X,y] = build_hog_from_folders(posDir, negDir, baseWin, cellSize)
% Fallback: build HOG features from pos/neg folders (simple and robust).
pos = [dir(fullfile(posDir,'*.jpg')); dir(fullfile(posDir,'*.png'))];
neg = [dir(fullfile(negDir,'*.jpg')); dir(fullfile(negDir,'*.png'))];
assert(~isempty(pos) && ~isempty(neg), 'No pos/neg images found in %s / %s', posDir, negDir);

N = numel(pos) + numel(neg);
% Determine feature length from one sample
tmpI = imread(fullfile(pos(1).folder, pos(1).name));
tmpI = im2uint8(tmpI);
tmpI = prepare_window(tmpI, baseWin);
[f, ~] = extractHOGFeatures(tmpI, 'CellSize', cellSize);
d = numel(f);

X = zeros(N, d, 'single');
y = zeros(N, 1);
idx = 1;

for i = 1:numel(pos)
    I = im2uint8(imread(fullfile(pos(i).folder, pos(i).name)));
    I = prepare_window(I, baseWin);
    X(idx,:) = extractHOGFeatures(I, 'CellSize', cellSize);
    y(idx) = 1; idx = idx+1;
end
for i = 1:numel(neg)
    I = im2uint8(imread(fullfile(neg(i).folder, neg(i).name)));
    I = prepare_window(I, baseWin);
    X(idx,:) = extractHOGFeatures(I, 'CellSize', cellSize);
    y(idx) = 0; idx = idx+1;
end
end

function [tp,fp,fn] = binary_counts(ytrue, yhat)
% ytrue, yhat in {0,1}
tp = sum(ytrue==1 & yhat==1);
fp = sum(ytrue==0 & yhat==1);
fn = sum(ytrue==1 & yhat==0);
end

function labels = C2str(Cs)
labels = arrayfun(@(c) sprintf('%.2g', c), Cs, 'UniformOutput', false);
end
