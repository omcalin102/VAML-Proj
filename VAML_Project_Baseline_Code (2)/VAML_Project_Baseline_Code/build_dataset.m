function [X, y] = build_dataset(posDir, negDir, varargin)
%BUILD_DATASET Load positive/negative crops and return a feature matrix.
%   [X, y] = BUILD_DATASET(posDir, negDir) reads all images from the
%   provided positive and negative directories, extracts descriptors using
%   extract_features, and returns a feature matrix X (N x D) with labels
%   y (N x 1) where positives are 1 and negatives are 0.
%
%   Optional name/value parameters (passed to extract_features):
%       'FeatureType'  'hog'     - 'hog', 'hog_pca', 'raw', 'edges'
%       'ResizeTo'     [64 128]  - width/height to resize each crop
%       'CellSize'     [8 8]
%       'BlockSize'    [2 2]
%       'BlockOverlap' [1 1]
%       'NumBins'      9
%       'EdgeMethod'   'Canny'   - edge detector for 'edges'
%       'PCA'          []        - struct with .Coeff/.Mu for hog_pca
%       'PCADim'       []        - optional output dimension for hog_pca
%
%   Additional options:
%       'Verbose'      true      - print progress logs
%
%   This helper centralises dataset creation so training/validation/demo
%   scripts stay concise and consistent across feature types.

p = inputParser;
addParameter(p, 'FeatureType', 'hog');
addParameter(p, 'ResizeTo', [64 128]);
addParameter(p, 'CellSize', [8 8]);
addParameter(p, 'BlockSize', [2 2]);
addParameter(p, 'BlockOverlap', [1 1]);
addParameter(p, 'NumBins', 9);
addParameter(p, 'EdgeMethod', 'Canny');
addParameter(p, 'PCA', []);
addParameter(p, 'PCADim', []);
addParameter(p, 'Verbose', true);
parse(p, varargin{:});
a = p.Results;

% Resolve directories to absolute paths to avoid label misassignment when
% callers pass relative folders and list_images returns absolute paths.
posDirAbs = abs_path(posDir);
negDirAbs = abs_path(negDir);

% Collect image paths
posFiles = list_images(posDirAbs);
negFiles = list_images(negDirAbs);

% Precompute labels based on directory of origin (more reliable than string
% matching against absolute paths which can differ if symlinks/relative
% paths are used by the caller)
labels = [ones(numel(posFiles),1); zeros(numel(negFiles),1)];

if a.Verbose
    fprintf('  - positives: %d | negatives: %d\n', numel(posFiles), numel(negFiles));
end

% Preallocate feature matrix (determine D from first image)
allFiles = [posFiles; negFiles];
if isempty(allFiles)
    X = zeros(0,0,'single'); y = zeros(0,1); return;
end

firstFeat = extract_features(imread(allFiles{1}), ...
    'FeatureType',a.FeatureType, 'ResizeTo',a.ResizeTo, 'CellSize',a.CellSize, ...
    'BlockSize',a.BlockSize, 'BlockOverlap',a.BlockOverlap, 'NumBins',a.NumBins, ...
    'EdgeMethod',a.EdgeMethod, 'PCA',a.PCA, 'PCADim',a.PCADim);
D = numel(firstFeat);
N = numel(allFiles);
X = zeros(N, D, 'single');
y = zeros(N, 1);
X(1,:) = firstFeat;
y(1) = labels(1);

% Process remaining files
timerStart = tic;
for k = 2:N
    X(k,:) = extract_features(imread(allFiles{k}), ...
        'FeatureType',a.FeatureType, 'ResizeTo',a.ResizeTo, 'CellSize',a.CellSize, ...
        'BlockSize',a.BlockSize, 'BlockOverlap',a.BlockOverlap, 'NumBins',a.NumBins, ...
        'EdgeMethod',a.EdgeMethod, 'PCA',a.PCA, 'PCADim',a.PCADim);
    y(k) = labels(k);

    if a.Verbose && (mod(k, 100) == 0 || k == N)
        progress_bar(k, N, timerStart, '  - extracting features');
    end
end

if a.Verbose
    fprintf('  - done. X: %d x %d | pos=%d / neg=%d | total %.1fs\n', ...
        size(X,1), size(X,2), sum(y==1), sum(y==0), toc(timerStart));
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
fpathAbs = abs_path(fpath);
posRoot  = ensure_trailing_filesep(abs_path(posDir));
lbl = startsWith(fpathAbs, posRoot);
end

function d = abs_path(p)
if isempty(p), d = ''; return; end
p = char(p);
if is_absolute(p)
    d = p;
else
    d = fullfile(pwd, p);
end
end

function tf = is_absolute(p)
if isempty(p)
    tf = false; return;
end
if ispc
    tf = startsWith(p, '\\') || (~isempty(p) && numel(p) >= 3 && p(2) == ':' && (p(3)=='\' || p(3)=='/'));
else
    tf = startsWith(p, filesep);
end
end

function d = ensure_trailing_filesep(d)
if ~isempty(d) && d(end) ~= filesep
    d = [d filesep];
end
end
