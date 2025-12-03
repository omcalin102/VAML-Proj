function model = train_model(X, y, modelType, paramValue, varargin)
%TRAIN_MODEL Dispatch to the requested learning algorithm.
%   MODEL = TRAIN_MODEL(X, y, modelType, paramValue) supports:
%       - 'svm' : paramValue == C (box constraint)
%       - 'knn' : paramValue == K (num neighbours)

switch lower(string(modelType))
    case "svm"
        model = train_svm(X, y, paramValue, varargin{:});
    case "knn"
        model = train_knn(X, y, paramValue, varargin{:});
    otherwise
        error('Unsupported model type: %s', modelType);
end
end
