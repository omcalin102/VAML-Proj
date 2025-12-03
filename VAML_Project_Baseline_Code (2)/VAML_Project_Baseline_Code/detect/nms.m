function [keepB, keepS, keepIdx] = nms(boxes, scores, iouThr, varargin)
% Greedy Non-Max Suppression on [x y w h] boxes by descending score.

if nargin<3 || isempty(iouThr), iouThr = 0.3; end  % IoU threshold (changeable)

if isempty(boxes)
    keepB = zeros(0,4); keepS = zeros(0,1); keepIdx = [];
    return;
end

% sort by score
[sortedS, order] = sort(scores(:), 'descend');
B = boxes(order, :);

x1 = B(:,1); y1 = B(:,2);
x2 = B(:,1)+B(:,3)-1; y2 = B(:,2)+B(:,4)-1;
areas = (x2 - x1 + 1) .* (y2 - y1 + 1);

keep = false(size(B,1),1);
while ~isempty(B)
    keep(1) = true;
    if size(B,1) == 1, break; end

    xx1 = max(B(1,1), B(2:end,1));
    yy1 = max(B(1,2), B(2:end,2));
    xx2 = min(B(1,1)+B(1,3)-1, B(2:end,1)+B(2:end,3)-1);
    yy2 = min(B(1,2)+B(1,4)-1, B(2:end,2)+B(2:end,4)-1);

    w = max(0, xx2 - xx1 + 1);
    h = max(0, yy2 - yy1 + 1);
    inter = w .* h;

    area1 = areas(1);
    area2 = areas(2:end);
    iou = inter ./ (area1 + area2 - inter);

    keepIdxLocal = find([true; iou <= iouThr]);     % IoU threshold (changeable)
    B     = B(keepIdxLocal, :);
    areas = areas(keepIdxLocal);
    sortedS = sortedS(keepIdxLocal);
    keep = keep(keepIdxLocal);
end

keepIdx = order(keep);
keepB   = boxes(keepIdx, :);
keepS   = scores(keepIdx);
end
