function model = train_model(X, y, modelType, paramValue, varargin)
%TRAIN_MODEL Dispatch to the requested learning algorithm.
%   MODEL = TRAIN_MODEL(X, y, modelType, paramValue) supports:
%       - 'svm'        : paramValue == C (box constraint)
%       - 'knn'        : paramValue == K (num neighbours)
%       - 'bag'        : paramValue == number of trees (bagging)
%       - 'boost'      : paramValue == number of trees (AdaBoost)
%       - 'rf'         : paramValue == number of trees (random forest)
%       - 'nn'/'dnn'   : paramValue == hidden layer width (fitcnet)

switch lower(string(modelType))
    case "svm"
        model = train_svm(X, y, paramValue, varargin{:});
    case "knn"
        model = train_knn(X, y, paramValue, varargin{:});
    case {"bag","bagging"}
        model = train_bagging(X, y, paramValue, varargin{:});
    case {"boost","ada"}
        model = train_boosting(X, y, paramValue, varargin{:});
    case {"rf","forest","randomforest"}
        model = train_random_forest(X, y, paramValue, varargin{:});
    case {"nn","dnn"}
        model = train_nn(X, y, paramValue, varargin{:});
    otherwise
        error('Unsupported model type: %s', modelType);
end
end
