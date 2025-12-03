function demo_feature_classifier_grid()
% Run feature x classifier grid with CV tables/plots for report justification.

clc; close all; rng(42,'twister');

% ---- PATHS ----
posDir   = fullfile('data','images','pos');
negDir   = fullfile('data','images','neg');
outTableDir  = fullfile('results','tables');
outFigureDir = fullfile('report','figs');
ensure_dir(outTableDir, outFigureDir);

% ---- CV SETTINGS ----
splitMode = 'holdout';    % 'holdout' or 'kfold'
holdout   = 0.2;          % holdout fraction if splitMode='holdout'
kfoldK    = 5;            % folds if splitMode='kfold'
primary   = 'F1';         % primary metric for sorting/plots

% ---- FEATURE DEFINITIONS ----
ResizeTo     = [64 128];
CellSize     = [8 8];
BlockSize    = [2 2];
BlockOverlap = [1 1];
NumBins      = 9;
HogPcaDim    = 128;       % PCs to keep for HOG+PCA
HogPcaVar    = 0.95;      % variance target (used if higher than HogPcaDim)

featConfigs = {
    struct('Name','HOG','Type','hog','Opts',struct()), ...
    struct('Name','HOG+PCA','Type','hog_pca','Opts',struct('PCADim',HogPcaDim)), ...
    struct('Name','Raw','Type','raw','Opts',struct('ResizeTo',[64 128])), ...
    struct('Name','Edges','Type','edges','Opts',struct())
};

% ---- CLASSIFIER DEFINITIONS ----
classifiers = {
    struct('Name','SVM','ParamName','C','Params',[0.1 0.3 1 3 10], ...
           'TrainFcn',@(X,y,p) train_svm(X,y,p,'Standardize',true), 'PredictFcn',@predict), ...
    struct('Name','kNN','ParamName','K','Params',[3 5 9 15], ...
           'TrainFcn',@(X,y,p) train_knn(X,y,p,'Standardize',true), 'PredictFcn',@predict), ...
    struct('Name','NN','ParamName','Hidden','Params',[16 32 64], ...
           'TrainFcn',@(X,y,p) train_nn(X,y,p,'Standardize',true,'Lambda',1e-4), 'PredictFcn',@predict)
};

% ---- BUILD DATASETS ----
datasets = struct('Name',{},'X',{},'y',{});
baseHog = [];
for i = 1:numel(featConfigs)
    cfg = featConfigs{i};
    fprintf('[1/%d] Building %s features ...\n', numel(featConfigs), cfg.Name);
    switch cfg.Type
        case 'hog'
            [X,y] = build_dataset(posDir, negDir, 'FeatureType','hog', ...
                'ResizeTo',ResizeTo,'CellSize',CellSize,'BlockSize',BlockSize, ...
                'BlockOverlap',BlockOverlap,'NumBins',NumBins);
            baseHog = X;
        case 'hog_pca'
            if isempty(baseHog)
                [baseHog,y] = build_dataset(posDir, negDir, 'FeatureType','hog', ...
                    'ResizeTo',ResizeTo,'CellSize',CellSize,'BlockSize',BlockSize, ...
                    'BlockOverlap',BlockOverlap,'NumBins',NumBins);
            end
            pcaModel = fit_pca(baseHog, 'NumComponents', HogPcaDim, 'VarianceToKeep', HogPcaVar);
            X = apply_pca_features(baseHog, pcaModel, HogPcaDim);
        case 'raw'
            [X,y] = build_dataset(posDir, negDir, 'FeatureType','raw', 'ResizeTo',ResizeTo);
        case 'edges'
            [X,y] = build_dataset(posDir, negDir, 'FeatureType','edges', 'ResizeTo',ResizeTo, 'EdgeMethod','Canny');
        otherwise
            error('Unknown feature type %s', cfg.Type);
    end
    datasets(end+1) = struct('Name',cfg.Name,'X',X,'y',y); %#ok<AGROW>
end

% ---- CROSS-VALIDATE ----
rows = {};
bestRows = {};
for d = 1:numel(datasets)
    ds = datasets(d);
    for c = 1:numel(classifiers)
        clf = classifiers{c};
        fprintf('  - CV %s + %s ...\n', ds.Name, clf.Name);
        cvArgs = {'PrimaryMetric',primary,'ParamName',clf.ParamName,'PredictFcn',clf.PredictFcn};
        switch splitMode
            case 'holdout'
                cvArgs = [cvArgs, {'Split','holdout','Holdout', holdout}];
            case 'kfold'
                cvArgs = [cvArgs, {'Split','kfold','K',kfoldK}];
        end
        cvRes = crossval_eval(ds.X, ds.y, clf.Params, cvArgs{:}, 'TrainFcn', clf.TrainFcn);
        for r = 1:size(cvRes,1)
            rows(end+1,:) = {ds.Name, clf.Name, cvRes(r,1), cvRes(r,2), cvRes(r,3), cvRes(r,4), cvRes(r,5)}; %#ok<AGROW>
        end
        bestRows(end+1,:) = {ds.Name, clf.Name, cvRes(1,1), cvRes(1,2), cvRes(1,3), cvRes(1,4), cvRes(1,5)}; %#ok<AGROW>
    end
end

T = cell2table(rows, 'VariableNames',{'Feature','Classifier','Param','Accuracy','Precision','Recall','F1'});
Best = cell2table(bestRows, 'VariableNames',{'Feature','Classifier','Param','Accuracy','Precision','Recall','F1'});

csvAll  = fullfile(outTableDir, 'feature_classifier_grid.csv');
csvBest = fullfile(outTableDir, 'feature_classifier_grid_best.csv');
writetable(T, csvAll);
writetable(Best, csvBest);
fprintf('Saved tables: %s and %s\n', csvAll, csvBest);

% ---- FIGURE ----
fig = figure('Color','w','Position',[100 100 960 420]);
cats = strcat(Best.Feature, ' + ', Best.Classifier, ' (', compose('%g', Best.Param), ')');
bar(Best.F1); grid on; box on;
set(gca,'XTick',1:numel(cats),'XTickLabel',cats,'XTickLabelRotation',30);
ylabel('F1'); ylim([0 1]);
title(sprintf('Best %s per Feature/Classifier', primary));
figPath = fullfile(outFigureDir,'feature_classifier_grid.png');
exportgraphics(fig, figPath, 'Resolution', 150);
close(fig);
fprintf('Saved figure: %s\n', figPath);

fprintf('DONE.\n');
end

function ensure_dir(varargin)
for i = 1:nargin
    d = varargin{i};
    if ~exist(d,'dir'), mkdir(d); end
end
end
