function P = imcrop_safe(I, box)
% Crop an image patch while clamping the box to valid pixel bounds.
x = max(1, box(1)); y = max(1, box(2)); w = max(1, box(3)); h = max(1, box(4));
x2 = min(size(I,2), x+w-1); y2 = min(size(I,1), y+h-1);
P = I(y:y2, x:x2);
end
