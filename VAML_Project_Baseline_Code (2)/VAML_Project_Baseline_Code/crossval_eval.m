function cvRes = crossval_eval(X, y, paramValues, varargin)
%CROSSVAL_EVAL Evaluate parameter grid with holdout or k-fold CV.
%   cvRes is a table sorted by the primary metric with columns:
%       <ParamName>, ModelType, Accuracy, Precision, Recall, F1

p = inputParser;
addParameter(p, 'Split', 'holdout');           % 'holdout' or 'kfold'
addParameter(p, 'Holdout', 0.2);               % fraction for holdout
addParameter(p, 'K', 5);                       % folds for kfold
addParameter(p, 'OutDir', '');                 % optional CSV output dir
addParameter(p, 'Label', '');                  % optional tag for filename
addParameter(p, 'PrimaryMetric', 'F1');        % for display/ordering only
addParameter(p, 'ModelType', 'svm');           % 'svm' or 'knn'
addParameter(p, 'ParamName', 'C');             % column header for grid
parse(p, varargin{:});
a = p.Results;

splitMode = lower(a.Split);
assert(any(strcmp(splitMode, {'holdout','kfold'})), 'Split must be holdout or kfold');

paramValues = paramValues(:)';
nP = numel(paramValues);

varNames = {a.ParamName, 'ModelType', 'Accuracy', 'Precision', 'Recall', 'F1'};
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
            preds = predict(mdl, Xva);
        case 'kfold'
            mdl = train_model(X, y, a.ModelType, param, 'Standardize', true);
            cvmdl = crossval(mdl, 'KFold', a.K);
            preds = kfoldPredict(cvmdl);
            Yva = y;
    end

    [acc, prec, rec, f1] = metrics_binary(preds, Yva);
    rows{i,1} = param;
    rows{i,2} = string(lower(a.ModelType));
    rows{i,3} = acc;
    rows{i,4} = prec;
    rows{i,5} = rec;
    rows{i,6} = f1;
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
metricMap = struct('accuracy',3,'precision',4,'recall',5,'f1',6);
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
