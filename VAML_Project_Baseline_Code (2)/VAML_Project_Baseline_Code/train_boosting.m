function model = train_boosting(X, y, numTrees, varargin)
%TRAIN_BOOSTING Train an AdaBoostM1 ensemble for binary/multi-class labels.
%   MODEL = TRAIN_BOOSTING(X, y, numTrees) fits an ensemble of shallow trees
%   using AdaBoost. Optional name/value pairs are passed to fitcensemble.

validateattributes(numTrees, {'numeric'}, {'scalar','positive','integer','finite'});

baseTree = templateTree('MaxNumSplits', 5);           % shallow trees for boosting
model = fitcensemble(X, y, 'Method','AdaBoostM1', ...
    'NumLearningCycles', numTrees, 'Learners', baseTree, varargin{:});
end
