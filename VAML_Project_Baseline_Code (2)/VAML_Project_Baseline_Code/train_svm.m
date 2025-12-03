function model = train_svm(X, y, C, varargin)
%TRAIN_SVM Convenience wrapper for training a linear SVM classifier.
%   MODEL = TRAIN_SVM(X, y, C) fits a linear SVM with box constraint C
%   using MATLAB's fitcsvm. Optional name/value pairs are forwarded to
%   fitcsvm so callers can control standardisation or class names.
%
%   Inputs:
%       X - N x D feature matrix
%       y - N x 1 labels (expects binary 0/1 but supports others)
%       C - box constraint (positive scalar)

validateattributes(C, {'numeric'}, {'scalar','positive','finite'});

% fitcsvm requires a unit box constraint when only a single class is
% present (one-class novelty detection). Detect this scenario up front so
% we can provide a clearer warning instead of surfacing MATLAB's runtime
% error.
if numel(unique(y)) == 1 && C ~= 1
    warning('train_svm:OneClassBoxConstraint', ...
        ['Only one class present in training data; overriding box ' ...
         'constraint to 1 for one-class SVM.']);
    C = 1;
end

args = {'KernelFunction','linear','BoxConstraint',C};
if ~any(strcmpi(varargin, 'Standardize'))
    args = [args, {'Standardize', true}];
end

model = fitcsvm(X, y, args{:}, varargin{:});
end
