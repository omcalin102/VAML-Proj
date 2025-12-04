function [keepB, keepS, keepIdx] = nms(boxes, scores, iouThr, varargin)
% Greedy Non-Max Suppression on [x y w h] boxes by descending score.

if nargin<3 || isempty(iouThr), iouThr = 0.3; end  % IoU threshold (changeable)

if isempty(boxes)
    keepB = zeros(0,4); keepS = zeros(0,1); keepIdx = [];
    return;
end

% sort by score
[~, order] = sort(scores(:), 'descend');
x1 = boxes(:,1); y1 = boxes(:,2);
x2 = boxes(:,1)+boxes(:,3)-1; y2 = boxes(:,2)+boxes(:,4)-1;
areas = (x2 - x1 + 1) .* (y2 - y1 + 1);

keepIdx = zeros(numel(order),1); k = 0;
while ~isempty(order)
    k = k + 1;
    i = order(1);               % index of highest remaining score
    keepIdx(k) = i;
    if numel(order) == 1, break; end

    % IoU of the top box vs. the rest
    xx1 = max(x1(i), x1(order(2:end)));
    yy1 = max(y1(i), y1(order(2:end)));
    xx2 = min(x2(i), x2(order(2:end)));
    yy2 = min(y2(i), y2(order(2:end)));

    w = max(0, xx2 - xx1 + 1);
    h = max(0, yy2 - yy1 + 1);
    inter = w .* h;
    iou = inter ./ (areas(i) + areas(order(2:end)) - inter);

    % strictly drop only when IoU exceeds the threshold so the loop shrinks
    order = order([true; iou <= iouThr]);  % keep first + low IoU
end

keepIdx = keepIdx(1:k);
keepB   = boxes(keepIdx, :);
keepS   = scores(keepIdx);
end
