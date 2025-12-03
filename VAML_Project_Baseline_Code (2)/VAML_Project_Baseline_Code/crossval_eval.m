function cvRes = crossval_eval(X, y, paramValues, varargin)
%CROSSVAL_EVAL Evaluate a hyperparameter grid with holdout or k-fold CV.
%   cvRes is N x 5 matrix: [param, Accuracy, Precision, Recall, F1].

p = inputParser;
addParameter(p, 'Split', 'holdout');           % 'holdout' or 'kfold'
addParameter(p, 'Holdout', 0.2);               % fraction for holdout
addParameter(p, 'K', 5);                       % folds for kfold
addParameter(p, 'OutDir', '');                 % optional CSV output dir
addParameter(p, 'PrimaryMetric', 'F1');        % for display/ordering only
addParameter(p, 'TrainFcn', @(Xtr,Ytr,p) train_svm(Xtr, Ytr, p, 'Standardize', true));
addParameter(p, 'PredictFcn', @predict);
addParameter(p, 'ParamName', 'Param');
parse(p, varargin{:});
a = p.Results;

splitMode = lower(a.Split);
assert(any(strcmp(splitMode, {'holdout','kfold'})), 'Split must be holdout or kfold');

paramValues = paramValues(:)';
nParams = numel(paramValues);
cvRes = zeros(nParams, 5);

for i = 1:nParams
    pval = paramValues(i);
    switch splitMode
        case 'holdout'
            cvp = cvpartition(y, 'Holdout', a.Holdout);
            Xtr = X(training(cvp),:); Ytr = y(training(cvp));
            Xva = X(test(cvp),:);     Yva = y(test(cvp));
            mdl = a.TrainFcn(Xtr, Ytr, pval);
            preds = a.PredictFcn(mdl, Xva);
        case 'kfold'
            mdl = a.TrainFcn(X, y, pval);
            cvmdl = crossval(mdl, 'KFold', a.K);
            preds = kfoldPredict(cvmdl);
            Yva = y;
    end

    [acc, prec, rec, f1] = metrics_binary(preds, Yva);
    cvRes(i,:) = [pval, acc, prec, rec, f1];
end

% Optional save
if ~isempty(a.OutDir)
    if exist(a.OutDir,'dir')~=7, mkdir(a.OutDir); end
    T = array2table(cvRes, 'VariableNames', {a.ParamName,'Accuracy','Precision','Recall','F1'});
    fname = fullfile(a.OutDir, sprintf('cv_results_%s.csv', splitMode));
    writetable(T, fname);
end

% For convenience, sort descending by primary metric
metricMap = struct('accuracy',2,'precision',3,'recall',4,'f1',5);
key = lower(a.PrimaryMetric);
if isfield(metricMap, key)
    [~, order] = sort(cvRes(:, metricMap.(key)), 'descend');
    cvRes = cvRes(order, :);
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
