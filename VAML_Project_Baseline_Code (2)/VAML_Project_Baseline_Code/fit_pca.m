function pcaStruct = fit_pca(X, varargin)
%FIT_PCA Compute PCA model and return coeff/mu for later projections.
%   pcaStruct = FIT_PCA(X) returns a struct with Coeff, Mu, and Explained
%   fields. Additional options:
%       'NumComponents'   []    - number of PCs to keep (empty = all)
%       'VarianceToKeep'  []    - fraction (0-1) of variance to retain
%       'Centered'        true  - whether to mean-center (passed to pca)
%
%   If both NumComponents and VarianceToKeep are provided, the smaller of
%   the two targets is used. This helper keeps PCA training separate from
%   feature extraction so HOG+PCA grids can reuse the same model.

p = inputParser;
addParameter(p, 'NumComponents', []);
addParameter(p, 'VarianceToKeep', []);
addParameter(p, 'Centered', true);
parse(p, varargin{:});
args = p.Results;

% Run PCA; defer component selection until after explained is known
[coeff, ~, ~, ~, explained, mu] = pca(X, 'Centered', args.Centered);

k = size(coeff, 2);
if ~isempty(args.VarianceToKeep)
    target = args.VarianceToKeep;
    if target > 1, target = target / 100; end
    cumExp = cumsum(explained) / 100;
    hit = find(cumExp >= target, 1, 'first');
    if ~isempty(hit)
        k = min(k, hit);
    end
end

if ~isempty(args.NumComponents)
    k = min(k, args.NumComponents);
end

pcaStruct = struct('Coeff', coeff(:,1:k), 'Mu', mu, 'Explained', explained(1:k));
end
