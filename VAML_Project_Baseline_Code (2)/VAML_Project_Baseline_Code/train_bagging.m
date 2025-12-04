function model = train_bagging(X, y, numTrees, varargin)
%TRAIN_BAGGING Train a bagged ensemble of decision trees for classification.
%   MODEL = TRAIN_BAGGING(X, y, numTrees) uses fitcensemble with Method='Bag'
%   and the requested number of trees. Additional name/value pairs are
%   forwarded to fitcensemble.

validateattributes(numTrees, {'numeric'}, {'scalar','positive','integer','finite'});

baseTree = templateTree('NumVariablesToSample','all');
model = fitcensemble(X, y, 'Method','Bag', 'NumLearningCycles',numTrees, ...
    'Learners', baseTree, varargin{:});
end
