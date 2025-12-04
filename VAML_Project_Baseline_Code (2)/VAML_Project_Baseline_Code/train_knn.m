function mdl = train_knn(X, y, k, varargin)
%TRAIN_KNN Train a k-NN classifier with sensible defaults.
%   mdl = TRAIN_KNN(X, y, k) trains a k-NN model with k neighbors and
%   returns a ClassificationKNN model suitable for predict/kfoldPredict.
%
%   Optional name/value parameters are passed to fitcknn, e.g.:
%       'Distance'     'euclidean' (default)
%       'Standardize'  true
%       'NSMethod'     'kdtree' or 'exhaustive'
%
%   This helper mirrors train_svm so crossval_eval can swap classifiers by
%   passing a different TrainFcn handle.

if nargin < 3 || isempty(k)
    k = 5;
end

p = inputParser;
addParameter(p, 'Distance', 'euclidean');
addParameter(p, 'Standardize', true);
% MATLAB's fitcknn only accepts 'kdtree' or 'exhaustive' for NSMethod in
% some versions. Default to 'auto' for forward compatibility, but map it to
% a supported option before calling fitcknn so older releases continue to
% work.
addParameter(p, 'NSMethod', 'auto');
parse(p, varargin{:});
a = p.Results;

nsMethod = a.NSMethod;
if strcmpi(nsMethod, 'auto')
    nsMethod = 'exhaustive';
end

mdl = fitcknn(X, y, ...
    'NumNeighbors', k, ...
    'Distance', a.Distance, ...
    'Standardize', a.Standardize, ...
    'NSMethod', nsMethod);
end
