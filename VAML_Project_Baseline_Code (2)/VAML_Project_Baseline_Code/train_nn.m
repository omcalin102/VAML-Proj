function model = train_nn(X, y, width, varargin)
%TRAIN_NN Train a shallow neural network classifier using fitcnet.
%   MODEL = TRAIN_NN(X, y, width) fits a fully-connected classifier with a
%   single hidden layer of the requested width. Name/value pairs are passed
%   to fitcnet so callers can adjust regularisation or optimisation.

validateattributes(width, {'numeric'}, {'scalar','positive','integer','finite'});

model = fitcnet(X, y, 'LayerSizes', width, varargin{:});
end
