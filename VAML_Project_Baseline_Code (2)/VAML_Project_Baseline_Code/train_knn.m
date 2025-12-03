function model = train_knn(X, y, K, varargin)
%TRAIN_KNN Convenience wrapper for training a k-NN classifier.
%   MODEL = TRAIN_KNN(X, y, K) fits a k-NN classifier with K neighbours
%   using MATLAB's fitcknn. Optional name/value pairs are forwarded to
%   fitcknn so callers can control distance metrics or standardisation.

validateattributes(K, {'numeric'}, {'scalar','positive','integer','finite'});

args = {'NumNeighbors', K};
if ~any(strcmpi(varargin, 'Standardize'))
    args = [args, {'Standardize', true}];
end

model = fitcknn(X, y, args{:}, varargin{:});
end
