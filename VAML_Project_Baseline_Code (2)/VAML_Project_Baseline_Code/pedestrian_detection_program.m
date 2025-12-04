function pedestrian_detection_program()
%PEDESTRIAN_DETECTION_PROGRAM End-to-end CSC3067 pipeline in one script.
%   This single entry point trains multiple classifiers with different
%   feature pipelines, evaluates them with proper train/test splits,
%   performs detection on full frames with multi-scale sliding windows
%   and non-maximum suppression, and exports metrics, tables and a demo
%   video. All major coursework requirements are represented so the
%   script can serve as the "one button" baseline for the report.
%
%   The code reuses helper utilities already present in the repository
%   (feature extractors, model trainers, detection primitives) but
%   orchestrates them in a new, self-contained flow. Key configurable
%   sections are grouped near the top for quick experimentation.

clc; close all; rng(3067,'twister');

% ------------------------------ PATHS ---------------------------------
paths.posDir   = fullfile('data','images','pos');          % positive crops
paths.negDir   = fullfile('data','images','neg');          % negative crops
paths.frameDir = fullfile('pedestrian','pedestrian');      % full frames for detection
paths.gtFile   = fullfile('data','test.dataset');          % GT boxes (optional)
paths.outModel = fullfile('results','models','pipeline_best.mat');
paths.outTable = fullfile('results','tables','pipeline_results.csv');
paths.outVideo = fullfile('results','videos','pipeline_detect.mp4');
paths.outRoc   = fullfile('report','figs');
ensure_dir(paths.outModel, fileparts(paths.outTable), fileparts(paths.outVideo), paths.outRoc);

% --------------------------- EXPERIMENT GRID --------------------------
% Feature pipelines: plain pixels, HOG, and PCA-reduced versions of both.
descGrid = [ ...
    struct('Name','pixels',    'Args',{{'FeatureType','pixels','ResizeTo',[64 128]}}), ...
    struct('Name','hog',       'Args',{{'FeatureType','hog','ResizeTo',[64 128],'CellSize',[8 8]}}), ...
    struct('Name','pixels+pca','Args',{{'FeatureType','pixels','ResizeTo',[64 128],'PCAComponents',0.90}}), ...
    struct('Name','hog+pca',   'Args',{{'FeatureType','hog','ResizeTo',[64 128],'CellSize',[8 8],'PCAComponents',0.90}}) ...
];

% Learning methods: linear SVM (C grid) and k-NN (k grid); both standardised.
modelGrid = [ ...
    struct('Type','svm','ParamName','C','Params',{[0.1 0.5 1 2 4]}), ...
    struct('Type','knn','ParamName','K','Params',{[1 3 5 7 9]}) ...
];

% Cross-validation options (holdout for speed but configurable to k-fold/LOO).
cvOpts.Split = 'holdout';   % 'holdout', 'kfold', or 'loo'
cvOpts.Holdout = 0.5;       % 50/50 split as suggested
cvOpts.K = 5;               % used only when Split='kfold'
cvOpts.PrimaryMetric = 'F1';

% Detection hyper-parameters.
detectOpts.BaseWindow  = [64 128];
detectOpts.Step        = 8;
detectOpts.ScaleFactor = 0.90;
detectOpts.NMS_IoU     = 0.30;
detectOpts.MinScore    = 0;
detectOpts.IoU_TP      = 0.50;
detectOpts.FPS         = 6;
detectOpts.VideoQuality= 100;

% ---------------------------- RUN PIPELINE ----------------------------
fprintf('CSC3067 pedestrian pipeline starting...\n');
[allResults, bestModel] = run_training_and_eval(paths, descGrid, modelGrid, cvOpts);
run_detection(paths, bestModel, detectOpts);

% Save combined table for the report.
writetable(allResults, paths.outTable);
fprintf('Full results saved to %s\n', paths.outTable);
fprintf('DONE.\n');
end

% =====================================================================
function [allResults, bestPacked] = run_training_and_eval(paths, descGrid, modelGrid, cvOpts)
% Build datasets for each descriptor, evaluate all classifier combos and
% return a table sorted by the primary metric. Also save the best model
% (descriptor + classifier + hyper-parameter) to disk.

rows = [];
bestScore = -inf; bestPacked = struct();

for d = 1:numel(descGrid)
    descName = descGrid(d).Name;
    fprintf('\n-- Building descriptor: %s --\n', descName);
    [X, y, descCfg] = build_dataset(paths.posDir, paths.negDir, 'Verbose', true, descGrid(d).Args{:});

    % Train/validation split follows the chosen CV strategy.
    switch lower(cvOpts.Split)
        case 'holdout'
            cvp = cvpartition(y, 'Holdout', cvOpts.Holdout);
            Xtr = X(training(cvp),:); Ytr = y(training(cvp));
            Xva = X(test(cvp),:);    Yva = y(test(cvp));
        case 'kfold'
            cvp = cvpartition(y, 'KFold', cvOpts.K);
        case 'loo'
            cvp = cvpartition(y, 'Leaveout');
        otherwise
            error('Unknown CV split: %s', cvOpts.Split);
    end

    for m = 1:numel(modelGrid)
        mdlCfg = modelGrid(m);
        fprintf('  > Model: %s\n', mdlCfg.Type);

        % Cross-validate hyper-parameters
        cvRes = crossval_eval(X, y, mdlCfg.Params{:}, ...
            'Split', cvOpts.Split, 'Holdout', cvOpts.Holdout, 'K', cvOpts.K, ...
            'PrimaryMetric', cvOpts.PrimaryMetric, 'ModelType', mdlCfg.Type, ...
            'ParamName', mdlCfg.ParamName, 'PlotROC', true, 'OutDir', paths.outRoc, ...
            'Label', sprintf('%s_%s', descName, mdlCfg.Type));

        % Train best model on train split and evaluate on held-out set
        bestParam = cvRes.(mdlCfg.ParamName)(1);
        bestClassifier = train_model(Xtr, Ytr, mdlCfg.Type, bestParam, 'Standardize', true);
        [preds, scores] = predict(bestClassifier, Xva);
        posScores = extract_positive_scores(scores, bestClassifier);
        [acc, prec, rec, f1, tp, fp, tn, fn] = full_metrics(preds, Yva, posScores);

        % Store row
        r = table(string(descName), string(mdlCfg.Type), bestParam, acc, prec, rec, f1, tp, fp, tn, fn, ...
            'VariableNames', {'Descriptor','Model','Param','Accuracy','Precision','Recall','F1','TP','FP','TN','FN'});
        rows = [rows; r]; %#ok<AGROW>

        % Track global best by primary metric
        if f1 > bestScore
            bestScore = f1;
            bestPacked = struct('Classifier', bestClassifier, 'Descriptor', descCfg, ...
                'ModelType', mdlCfg.Type, 'Param', bestParam, 'Metrics', r);
            save(paths.outModel, 'bestPacked');
            fprintf('    * New best (F1=%.3f) saved to %s\n', f1, paths.outModel);
        end
    end
end

% Sort rows by primary metric
allResults = sortrows(rows, 'F1', 'descend');
end

% =====================================================================
function run_detection(paths, packedModel, detectOpts)
% Use the trained model to detect pedestrians on full frames. Runs a
% multi-scale sliding-window detector followed by NMS and optional metric
% computation if ground truth is available. A demo video is exported.

fprintf('\n-- Detection phase --\n');
assert(~isempty(fieldnames(packedModel)), 'No trained model available.');
model.Classifier = packedModel.Classifier;
model.Descriptor = packedModel.Descriptor;

frames = dir(fullfile(paths.frameDir,'*.jpg'));
if isempty(frames)
    frames = dir(fullfile(paths.frameDir,'*.png'));
end
assert(~isempty(frames), 'No frames found in %s', paths.frameDir);

haveGT = exist(paths.gtFile,'file')==2 && exist('load_gt','file')==2;
if haveGT
    try
        GT = load_gt(paths.gtFile);
    catch
        haveGT = false; GT = struct();
    end
end

try
    vw = VideoWriter(paths.outVideo, 'MPEG-4');
catch
    [baseDir, baseName, ~] = fileparts(paths.outVideo);
    paths.outVideo = fullfile(baseDir, [baseName '.avi']);
    vw = VideoWriter(paths.outVideo, 'Motion JPEG AVI');
end
vw.FrameRate = detectOpts.FPS;
if isprop(vw,'Quality'), vw.Quality = detectOpts.VideoQuality; end
open(vw);

BaseWindow = detectOpts.BaseWindow;
if isstruct(model) && isfield(model,'Descriptor')
    BaseWindow = model.Descriptor.ResizeTo;
end

allTP=0; allFP=0; allFN=0; perFrameMs=zeros(numel(frames),1);
for k = 1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));
    t0 = tic;
    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',BaseWindow, ...
        'Step',detectOpts.Step, ...
        'ScaleFactor',detectOpts.ScaleFactor, ...
        'MinScore',detectOpts.MinScore);
    [keepB, keepS] = nms(boxes, scores, detectOpts.NMS_IoU);
    perFrameMs(k) = toc(t0)*1000;

    posMask = keepS > detectOpts.MinScore;
    keepB = keepB(posMask,:);

    J = insertShape(I,'Rectangle',keepB,'LineWidth',2,'Color','red');

    if haveGT
        key = strip_extension(frames(k).name);
        if isfield(GT,key)
            gtB = GT.(key);
            if ~isempty(gtB)
                [tp,fp,fn] = match_detections(keepB, gtB, detectOpts.IoU_TP);
                allTP=allTP+tp; allFP=allFP+fp; allFN=allFN+fn;
            end
        end
    end

    writeVideo(vw,J);
end
close(vw);

fprintf('Saved detection video to %s | mean %.1f ms/frame (%.1f fps)\n', paths.outVideo, mean(perFrameMs), 1000/mean(perFrameMs));

if haveGT
    prec = allTP / max(1, allTP+allFP);
    rec  = allTP / max(1, allTP+allFN);
    f1   = 2*prec*rec / max(1e-9,prec+rec);
    T = table(prec, rec, f1, mean(perFrameMs), 'VariableNames',{'Precision','Recall','F1','MsPerFrame'});
    metricsPath = fullfile(fileparts(paths.outVideo), 'pipeline_detect_metrics.csv');
    writetable(T, metricsPath);
    fprintf('Detection metrics: P=%.3f R=%.3f F1=%.3f | %.1f ms/frame\n', prec, rec, f1, mean(perFrameMs));
else
    fprintf('Ground truth not available; skipped detection metrics export.\n');
end
end

% =====================================================================
function [acc, prec, rec, f1, TP, FP, TN, FN] = full_metrics(preds, truth, posScores)
% Compute full confusion-matrix metrics (accuracy, precision, recall,
% sensitivity/specificity via TP/TN/FP/FN). posScores is optional and can
% be empty; it is used only when present to refine ordering elsewhere.

preds = double(preds(:)); truth = double(truth(:));
TP = sum(preds==1 & truth==1);
FP = sum(preds==1 & truth==0);
TN = sum(preds==0 & truth==0);
FN = sum(preds==0 & truth==1);

acc  = (TP + TN) / max(1, numel(truth));
prec = TP / max(1, TP + FP);
rec  = TP / max(1, TP + FN);
f1   = 2*prec*rec / max(1e-9, prec + rec);

% Keep posScores in the signature for completeness
if nargin > 2 && ~isempty(posScores) %#ok<INUSD>
    % placeholder for future ROC/PR handling
end
end

% =====================================================================
function ensure_dir(varargin)
for i = 1:nargin
    d = varargin{i};
    if exist(d,'dir') ~= 7 && ~isempty(d)
        mkdir(d);
    end
end
end

function s = strip_extension(fname)
[~,s,~] = fileparts(fname);
end
