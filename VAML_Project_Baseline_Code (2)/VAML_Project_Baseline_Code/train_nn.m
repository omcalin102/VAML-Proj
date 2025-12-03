function mdl = train_nn(X, y, hiddenUnits, varargin)
%TRAIN_NN Train a simple feedforward neural net classifier (fitcnet wrapper).
%   mdl = TRAIN_NN(X, y, hiddenUnits) fits a network with the specified
%   number of hidden units (scalar). Optional name/value parameters are
%   forwarded to fitcnet, e.g.:
%       'Activations'   'relu' | 'tanh'
%       'Standardize'   true/false (default: true)
%       'Lambda'        L2 regularization (default: 1e-4)
%
%   This mirrors train_svm/train_knn so crossval_eval can sweep NN configs.

if nargin < 3 || isempty(hiddenUnits)
    hiddenUnits = 32;
end

p = inputParser;
addParameter(p, 'Activations', 'relu');
addParameter(p, 'Standardize', true);
addParameter(p, 'Lambda', 1e-4);
parse(p, varargin{:});
a = p.Results;

mdl = fitcnet(X, y, ...
    'LayerSizes', hiddenUnits, ...
    'Activations', a.Activations, ...
    'Standardize', a.Standardize, ...
    'Lambda', a.Lambda);
end
