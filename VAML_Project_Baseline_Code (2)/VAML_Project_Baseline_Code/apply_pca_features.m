function Y = apply_pca_features(X, pcaStruct, k)
%APPLY_PCA_FEATURES Project feature vectors using a precomputed PCA model.
%   Y = APPLY_PCA_FEATURES(X, pcaStruct) subtracts the PCA mean and projects
%   the rows of X using pcaStruct.Coeff.
%
%   Y = APPLY_PCA_FEATURES(X, pcaStruct, k) keeps only the first k
%   components (or all available if k is empty/omitted).
%
%   pcaStruct must contain fields:
%       - Coeff : PCA coefficients (D x Dp)
%       - Mu    : mean vector (1 x D)
%
%   This helper mirrors the PCA path in extract_features but is reusable for
%   dataset-level projections (e.g., HOG+PCA sweeps).

if nargin < 3
    k = [];
end

if isempty(pcaStruct) || ~isfield(pcaStruct, 'Coeff') || ~isfield(pcaStruct, 'Mu')
    error('pcaStruct must contain fields Coeff and Mu.');
end

coeff = pcaStruct.Coeff;
mu    = pcaStruct.Mu;

if isempty(k)
    k = size(coeff, 2);
end
k = min(k, size(coeff,2));

% Ensure rows are projected; bsxfun handles both row and column means
Y = bsxfun(@minus, X, mu) * coeff(:,1:k);
end
