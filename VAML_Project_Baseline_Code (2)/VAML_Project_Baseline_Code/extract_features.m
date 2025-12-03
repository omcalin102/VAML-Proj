function feat = extract_features(img, varargin)
%EXTRACT_FEATURES Unified feature extractor for multiple descriptors.
%   feat = EXTRACT_FEATURES(img) computes HOG features for the input image.
%
%   Name/value parameters:
%       'FeatureType'  'hog'     - 'hog', 'hog_pca', 'raw', 'edges'
%       'ResizeTo'     [64 128]  - target [w h] for resizing
%       'CellSize'     [8 8]
%       'BlockSize'    [2 2]
%       'BlockOverlap' [1 1]
%       'NumBins'      9
%       'EdgeMethod'   'Canny'   - edge detector for 'edges'
%       'PCA'          []        - struct with fields .Coeff and .Mu
%       'PCADim'       []        - optional output dimension for hog_pca
%
%   This helper keeps feature extraction consistent across experiment
%   scripts so feature/classifier sweeps can share the same call surface.

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
parse(p, varargin{:});
a = p.Results;

% Ensure grayscale single images for non-HOG paths
if size(img,3) > 1
    imgGray = rgb2gray(img);
else
    imgGray = img;
end
imgGray = im2single(imgGray);
imgResized = imresize(imgGray, a.ResizeTo);

switch lower(a.FeatureType)
    case 'hog'
        feat = extract_hog(img, 'ResizeTo',a.ResizeTo, 'CellSize',a.CellSize, ...
            'BlockSize',a.BlockSize, 'BlockOverlap',a.BlockOverlap, 'NumBins',a.NumBins);
    case 'hog_pca'
        base = extract_hog(img, 'ResizeTo',a.ResizeTo, 'CellSize',a.CellSize, ...
            'BlockSize',a.BlockSize, 'BlockOverlap',a.BlockOverlap, 'NumBins',a.NumBins);
        if isempty(a.PCA)
            warning('PCA struct missing; returning raw HOG features.');
            feat = base;
        else
            feat = apply_pca_features(base, a.PCA, a.PCADim);
        end
    case 'raw'
        feat = imgResized(:)';
    case 'edges'
        edges = edge(imgResized, a.EdgeMethod);
        feat = im2single(edges(:))';
    otherwise
        error('Unknown FeatureType: %s', a.FeatureType);
end

feat = single(feat);
end
