function cvRes = crossval_eval(X, y, paramValues, varargin)
%CROSSVAL_EVAL Evaluate parameter grid with holdout/k-fold/LOO CV + ROC.
%   cvRes is a table sorted by the primary metric with columns:
%       <ParamName>, ModelType, Accuracy, Precision, Recall, F1, AUC

p = inputParser;
addParameter(p, 'Split', 'holdout');           % 'holdout' or 'kfold'
addParameter(p, 'Holdout', 0.2);               % fraction for holdout
addParameter(p, 'K', 5);                       % folds for kfold
addParameter(p, 'OutDir', '');                 % optional CSV output dir
addParameter(p, 'Label', '');                  % optional tag for filename
addParameter(p, 'PrimaryMetric', 'F1');        % for display/ordering only
addParameter(p, 'ModelType', 'svm');           % 'svm' or 'knn'
addParameter(p, 'ParamName', 'C');             % column header for grid
addParameter(p, 'PlotROC', false);             % generate ROC curves when true
parse(p, varargin{:});
a = p.Results;

splitMode = lower(a.Split);
assert(any(strcmp(splitMode, {'holdout','kfold','loo'})), 'Split must be holdout, kfold or loo');

paramValues = paramValues(:)';
nP = numel(paramValues);

varNames = {a.ParamName, 'ModelType', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'};
rows = cell(nP, numel(varNames));

for i = 1:nP
    if iscell(paramValues)
        param = paramValues{i};
    else
        param = paramValues(i);
    end
    switch splitMode
        case 'holdout'
            cvp = cvpartition(y, 'Holdout', a.Holdout);
            Xtr = X(training(cvp),:); Ytr = y(training(cvp));
            Xva = X(test(cvp),:);     Yva = y(test(cvp));
            mdl = train_model(Xtr, Ytr, a.ModelType, param, 'Standardize', true);
            [preds, scores] = predict(mdl, Xva);
        case 'kfold'
            mdl = train_model(X, y, a.ModelType, param, 'Standardize', true);
            cvmdl = crossval(mdl, 'KFold', a.K);
            [preds, scores] = kfoldPredict(cvmdl);
            Yva = y;
        case 'loo'
            mdl = train_model(X, y, a.ModelType, param, 'Standardize', true);
            cvp = cvpartition(y, 'Leaveout');
            cvmdl = crossval(mdl, 'CVPartition', cvp);
            [preds, scores] = kfoldPredict(cvmdl);
            Yva = y;
    end

    posScores = extract_positive_scores(scores, mdl);
    [acc, prec, rec, f1] = metrics_binary(preds, Yva);
    auc = compute_auc(posScores, Yva);
    if a.PlotROC && ~isempty(a.OutDir)
        ensure_dir(a.OutDir);
        label = sprintf('%s_%s_%s', splitMode, lower(a.ModelType), string(param));
        rocPath = fullfile(a.OutDir, sprintf('roc_%s.png', label));
        plot_roc_curve(posScores, Yva, auc, rocPath, label);
    end
    rows{i,1} = param;
    rows{i,2} = string(lower(a.ModelType));
    rows{i,3} = acc;
    rows{i,4} = prec;
    rows{i,5} = rec;
    rows{i,6} = f1;
    rows{i,7} = auc;
end

cvRes = cell2table(rows, 'VariableNames', varNames);
cvRes.ModelType = string(cvRes.ModelType);

% Optional save
if ~isempty(a.OutDir)
    if exist(a.OutDir,'dir')~=7, mkdir(a.OutDir); end
    label = a.Label;
    if strlength(label) > 0
        label = ['_' char(label)];
    end
    fname = fullfile(a.OutDir, sprintf('cv_results_%s%s.csv', splitMode, label));
    writetable(cvRes, fname);
end

% For convenience, sort descending by primary metric
metricMap = struct('accuracy',3,'precision',4,'recall',5,'f1',6,'auc',7);
key = lower(a.PrimaryMetric);
if isfield(metricMap, key)
    cvRes = sortrows(cvRes, metricMap.(key), 'descend');
end
end

% ---- helpers ----
function [acc, prec, rec, f1] = metrics_binary(preds, truth)
preds = double(preds(:)); truth = double(truth(:));
if numel(preds) ~= numel(truth)
    error('Prediction and truth vectors must match.');
end

TP = sum(preds==1 & truth==1);
FP = sum(preds==1 & truth==0);
TN = sum(preds==0 & truth==0);
FN = sum(preds==0 & truth==1);

acc  = (TP + TN) / max(1, numel(truth));
prec = TP / max(1, TP + FP);
rec  = TP / max(1, TP + FN);
f1   = 2 * prec * rec / max(1e-9, prec + rec);
end

function sc = extract_positive_scores(scores, mdl)
if nargin < 2 || isempty(scores)
    sc = [];
    return;
end

if isstruct(mdl) && isfield(mdl,'Classifier')
    cls = mdl.Classifier;
else
    cls = mdl;
end

if isempty(scores) || size(scores,2) < 2
    sc = scores(:);
    return;
end

names = string(cls.ClassNames);
posIx = find(names=="1" | lower(names)=="pos" | lower(names)=="positive" | names=="true", 1);
if isempty(posIx)
    posIx = size(scores,2);
end
sc = scores(:,posIx);
end

function auc = compute_auc(scores, truth)
auc = NaN;
if isempty(scores)
    return;
end
try
    [~,~,~,auc] = perfcurve(truth, scores, 1);
catch
    auc = NaN;
end
end

function plot_roc_curve(scores, truth, auc, outPath, label)
try
    [fpr, tpr, ~, ~] = perfcurve(truth, scores, 1);
    figure('Visible','off');
    plot(fpr, tpr, 'LineWidth', 2);
    grid on; xlim([0 1]); ylim([0 1]);
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC: %s (AUC=%.3f)', label, auc));
    saveas(gcf, outPath);
    close(gcf);
catch
    % Ignore plotting errors (e.g., unavailable display)
end
end

function ensure_dir(d)
if exist(d,'dir')~=7, mkdir(d); end
end
