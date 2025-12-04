function feat = extract_hog_parts(I, varargin)
%EXTRACT_HOG_PARTS HOG descriptor that pools over vertical body parts.
%   feat = EXTRACT_HOG_PARTS(I, ...) splits the resized patch into top/mid/
%   bottom thirds, extracts HOG from each part, and concatenates the vectors
%   into a single descriptor. The resize step guarantees a fixed-length
%   feature regardless of input size and exposes the same tunable knobs as
%   extract_hog.
%
% Tunables (all optional):
%   'CellSize'     [8 8]
%   'BlockSize'    [2 2]          (in cells)
%   'BlockOverlap' [1 1]          (in cells)
%   'NumBins'      9
%   'ResizeTo'     [64 128]       (width × height for the full window)

p = inputParser;
addParameter(p, 'CellSize', [8 8]);
addParameter(p, 'BlockSize', [2 2]);
addParameter(p, 'BlockOverlap', [1 1]);
addParameter(p, 'NumBins', 9);
addParameter(p, 'ResizeTo', [64 128]);
parse(p, varargin{:});
args = p.Results;

validateattributes(args.ResizeTo, {'numeric'}, {'vector','numel',2,'positive','finite'});

% Normalize image
validateattributes(I, {'numeric','logical','uint8','uint16'}, {'nonempty'});
if size(I,3) > 1, I = rgb2gray(I); end
I = im2uint8(I);
I = imresize(I, [args.ResizeTo(2) args.ResizeTo(1)]);  % [h w]

% Split into three vertical parts
h = size(I,1);
partH = max(8, floor(h/3));
rows = {1:partH, partH+1:partH*2, partH*2+1:h};
targetH = [partH partH partH + (h - partH*2)];
featParts = cell(1,3);
for i = 1:3
    patch = I(rows{i},:);
    % Resize each part to a consistent height so HOG length is fixed
    patch = imresize(patch, [targetH(i) size(I,2)]);
    partFeat = extractHOGFeatures(patch, ...
        'CellSize',     args.CellSize, ...
        'BlockSize',    args.BlockSize, ...
        'BlockOverlap', args.BlockOverlap, ...
        'NumBins',      args.NumBins);
    featParts{i} = partFeat(:)';
end

feat = single([featParts{:}]);
if iscolumn(feat), feat = feat'; end
end
