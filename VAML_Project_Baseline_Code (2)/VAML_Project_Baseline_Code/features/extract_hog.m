function feat = extract_hog(I, varargin)
% EXTRACT_HOG  Return a 1×D HOG feature row vector for a patch.
% Example:
%   f = extract_hog(I, 'ResizeTo',[64 128]);  % width×height to FIX feature length
%
% Tunables (all optional):
%   'CellSize'     [8 8]
%   'BlockSize'    [2 2]          (in cells)
%   'BlockOverlap' [1 1]          (in cells)
%   'NumBins'      9
%   'ResizeTo'     [] or [w h]    (set to [64 128] for determinism)

p = inputParser;
addParameter(p, 'CellSize', [8 8]);
addParameter(p, 'BlockSize', [2 2]);
addParameter(p, 'BlockOverlap', [1 1]);
addParameter(p, 'NumBins', 9);
addParameter(p, 'ResizeTo', []);
parse(p, varargin{:});
args = p.Results;

% --- Validate and normalize image ---
validateattributes(I, {'numeric','logical','uint8','uint16'}, {'nonempty'});
if size(I,3) > 1, I = rgb2gray(I); end
I = im2uint8(I);

% Optional resize (recommended so D is constant across images)
if ~isempty(args.ResizeTo)
    validateattributes(args.ResizeTo, {'numeric'}, {'vector','numel',2,'positive','finite'});
    I = imresize(I, [args.ResizeTo(2) args.ResizeTo(1)]);   % [h w]
end

% --- HOG extraction ---
feat = extractHOGFeatures( ...
    I, ...
    'CellSize',     args.CellSize, ...
    'BlockSize',    args.BlockSize, ...
    'BlockOverlap', args.BlockOverlap, ...
    'NumBins',      args.NumBins);

% Ensure row vector
if iscolumn(feat), feat = feat'; end
feat = single(feat);
end
