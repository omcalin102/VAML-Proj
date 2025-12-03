function [X, y, descCfg] = build_dataset(posDir, negDir, varargin)
%BUILD_DATASET Load positive/negative crops and return feature matrix.
%   [X, y, descCfg] = BUILD_DATASET(posDir, negDir) reads all images from
%   the provided positive and negative directories, extracts descriptors
%   (HOG, raw pixels, optionally PCA-reduced) and returns a feature matrix
%   X (N x D) with labels y (N x 1) where positives are 1 and negatives are
%   0. descCfg captures the descriptor settings (including PCA transform)
%   so the detector can reuse the exact feature pipeline at test time.
%
%   Descriptor options (name/value pairs):
%       'FeatureType'   'hog'     - 'hog' or 'pixels'
%       'ResizeTo'      [64 128]  - width/height to resize each crop
%       'CellSize'      [8 8]     - HOG only
%       'BlockSize'     [2 2]     - HOG only
%       'BlockOverlap'  [1 1]     - HOG only
%       'NumBins'       9         - HOG only
%       'PCAComponents' []        - integer (#dims) or fraction (0-1) to
%                                   keep via PCA (applied after descriptor)
%
%   Additional options:
%       'Verbose'       true      - print progress logs
%
%   This helper centralises dataset creation so training/validation/demo
%   scripts stay concise and consistent while supporting multiple feature
%   descriptors required by the coursework spec.

p = inputParser;
addParameter(p, 'FeatureType', 'hog');
addParameter(p, 'ResizeTo', [64 128]);
addParameter(p, 'CellSize', [8 8]);
addParameter(p, 'BlockSize', [2 2]);
addParameter(p, 'BlockOverlap', [1 1]);
addParameter(p, 'NumBins', 9);
addParameter(p, 'PCAComponents', []);      % [] disables PCA
addParameter(p, 'Verbose', true);
parse(p, varargin{:});
a = p.Results;

descCfg = struct( ...
    'Type', lower(string(a.FeatureType)), ...
    'ResizeTo', a.ResizeTo, ...
    'CellSize', a.CellSize, ...
    'BlockSize', a.BlockSize, ...
    'BlockOverlap', a.BlockOverlap, ...
    'NumBins', a.NumBins, ...
    'PCA', []);  % filled later if PCA requested

% Collect image paths
posFiles = list_images(posDir);
negFiles = list_images(negDir);

if a.Verbose
    fprintf('  - positives: %d | negatives: %d\n', numel(posFiles), numel(negFiles));
end

% Preallocate feature matrix (determine D from first image)
allFiles = [posFiles; negFiles];
if isempty(allFiles)
    X = zeros(0,0,'single'); y = zeros(0,1); return;
end

firstFeat = extract_descriptor(imread(allFiles{1}), descCfg);
D = numel(firstFeat);
N = numel(allFiles);
X = zeros(N, D, 'single');
y = zeros(N, 1);
X(1,:) = firstFeat;
y(1) = label_from_path(allFiles{1}, posDir);

% Process remaining files
for k = 2:N
    if a.Verbose && mod(k, 200) == 0
        fprintf('  - features %4d/%4d\n', k, N); drawnow;
    end
    X(k,:) = extract_descriptor(imread(allFiles{k}), descCfg);
    y(k) = label_from_path(allFiles{k}, posDir);
end

% Optional PCA (after descriptor extraction)
if ~isempty(a.PCAComponents)
    comp = a.PCAComponents;
    if ~(isscalar(comp) && isnumeric(comp) && isfinite(comp) && comp > 0)
        error('PCAComponents must be a positive scalar (components or fraction in (0,1]).');
    end

    [coeff, score, ~, ~, explained, mu] = pca(double(X));

    if comp < 1                               % fraction of variance
        cumExp = cumsum(explained) / 100;
        k = find(cumExp >= comp, 1);
    else                                      % explicit dimension
        k = min(size(coeff,2), round(comp));
    end

    coeffK = coeff(:,1:k); explainedK = explained(1:k); %#ok<NASGU>
    X = single(score(:,1:k));                % PCA-projected dataset
    descCfg.PCA = struct('Coeff', single(coeffK), 'Mu', single(mu), ...
                         'NumComponents', k, 'Explained', single(explainedK));
    if a.Verbose
        fprintf('  - PCA kept %d dims (%.1f%% variance)\n', k, sum(explainedK));
    end
end

if a.Verbose
    fprintf('  - done. X: %d x %d | pos=%d / neg=%d\n', size(X,1), size(X,2), sum(y==1), sum(y==0));
end
end

% ---- helpers ----
function files = list_images(root)
if isempty(root) || exist(root,'dir')~=7
    files = {};
    return;
end
pat = fullfile(root, '*.*');
all = dir(pat);
all = all(~[all.isdir]);
keep = endsWith({all.name}, {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}, 'IgnoreCase', true);
all = all(keep);
files = fullfile({all.folder}, {all.name});
files = files(:);
end

function lbl = label_from_path(fpath, posDir)
if startsWith(string(fpath), string(posDir))
    lbl = 1;
else
    lbl = 0;
end
end
