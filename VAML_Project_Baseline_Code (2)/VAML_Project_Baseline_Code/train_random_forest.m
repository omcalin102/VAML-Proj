function model = train_random_forest(X, y, numTrees, varargin)
%TRAIN_RANDOM_FOREST Train a random forest classifier.
%   MODEL = TRAIN_RANDOM_FOREST(X, y, numTrees) uses bagging with
%   feature-subsampled decision trees. Additional name/value pairs are
%   forwarded to fitcensemble.

validateattributes(numTrees, {'numeric'}, {'scalar','positive','integer','finite'});

% "sqrt" feature subsampling mirrors the classic random forest recipe
baseTree = templateTree('NumVariablesToSample', 'sqrt');
model = fitcensemble(X, y, 'Method','Bag', 'NumLearningCycles', numTrees, ...
    'Learners', baseTree, varargin{:});
end
