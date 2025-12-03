function feat = extract_pixels(I, varargin)
%EXTRACT_PIXELS Flatten resized grayscale pixels into a 1xD vector.
%   feat = extract_pixels(I, 'ResizeTo', [w h]) converts image I to uint8
%   grayscale, resizes to [h w], and returns the flattened pixel values.

p = inputParser;
addParameter(p, 'ResizeTo', [64 128]);
parse(p, varargin{:});
a = p.Results;

if size(I,3) > 1, I = rgb2gray(I); end
I = im2uint8(I);
if ~isempty(a.ResizeTo)
    I = imresize(I, [a.ResizeTo(2) a.ResizeTo(1)]);
end

feat = double(I(:))';
feat = single(feat);
end
