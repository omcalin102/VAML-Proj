function cvRes = crossval_eval(X, y, C_values, varargin)
%CROSSVAL_EVAL Evaluate C grid with holdout or k-fold CV and return metrics.
%   cvRes is N x 5 matrix: [C, Accuracy, Precision, Recall, F1].

p = inputParser;
addParameter(p, 'Split', 'holdout');           % 'holdout' or 'kfold'
addParameter(p, 'Holdout', 0.2);               % fraction for holdout
addParameter(p, 'K', 5);                       % folds for kfold
addParameter(p, 'OutDir', '');                 % optional CSV output dir
addParameter(p, 'PrimaryMetric', 'F1');        % for display/ordering only
parse(p, varargin{:});
a = p.Results;

splitMode = lower(a.Split);
assert(any(strcmp(splitMode, {'holdout','kfold'})), 'Split must be holdout or kfold');

C_values = C_values(:)';
nC = numel(C_values);
cvRes = zeros(nC, 5);

for i = 1:nC
    C = C_values(i);
    switch splitMode
        case 'holdout'
            cvp = cvpartition(y, 'Holdout', a.Holdout);
            Xtr = X(training(cvp),:); Ytr = y(training(cvp));
            Xva = X(test(cvp),:);     Yva = y(test(cvp));
            mdl = train_svm(Xtr, Ytr, C, 'Standardize', true);
            preds = predict(mdl, Xva);
        case 'kfold'
            mdl = train_svm(X, y, C, 'Standardize', true);
            cvmdl = crossval(mdl, 'KFold', a.K);
            preds = kfoldPredict(cvmdl);
            Yva = y;
    end

    [acc, prec, rec, f1] = metrics_binary(preds, Yva);
    cvRes(i,:) = [C, acc, prec, rec, f1];
end

% Optional save
if ~isempty(a.OutDir)
    if exist(a.OutDir,'dir')~=7, mkdir(a.OutDir); end
    T = array2table(cvRes, 'VariableNames', {'C','Accuracy','Precision','Recall','F1'});
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
