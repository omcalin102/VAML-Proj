function [X, y] = build_dataset(posDir, negDir, varargin)
%BUILD_DATASET Load positive/negative crops and return HOG feature matrix.
%   [X, y] = BUILD_DATASET(posDir, negDir) reads all images from the
%   provided positive and negative directories, extracts HOG descriptors
%   using extract_hog, and returns a feature matrix X (N x D) with labels
%   y (N x 1) where positives are 1 and negatives are 0.
%
%   Optional name/value parameters (passed to extract_hog):
%       'ResizeTo'     [64 128]  - width/height to resize each crop
%       'CellSize'     [8 8]
%       'BlockSize'    [2 2]
%       'BlockOverlap' [1 1]
%       'NumBins'      9
%
%   Additional options:
%       'Verbose'      true      - print progress logs
%
%   This helper centralises dataset creation so training/validation/demo
%   scripts stay concise and consistent.

p = inputParser;
addParameter(p, 'ResizeTo', [64 128]);
addParameter(p, 'CellSize', [8 8]);
addParameter(p, 'BlockSize', [2 2]);
addParameter(p, 'BlockOverlap', [1 1]);
addParameter(p, 'NumBins', 9);
addParameter(p, 'Verbose', true);
parse(p, varargin{:});
a = p.Results;

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

firstFeat = extract_hog(imread(allFiles{1}), ...
    'ResizeTo',a.ResizeTo, 'CellSize',a.CellSize, ...
    'BlockSize',a.BlockSize, 'BlockOverlap',a.BlockOverlap, 'NumBins',a.NumBins);
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
    X(k,:) = extract_hog(imread(allFiles{k}), ...
        'ResizeTo',a.ResizeTo, 'CellSize',a.CellSize, ...
        'BlockSize',a.BlockSize, 'BlockOverlap',a.BlockOverlap, 'NumBins',a.NumBins);
    y(k) = label_from_path(allFiles{k}, posDir);
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
