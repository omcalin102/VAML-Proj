function [keepB, keepS, keepIdx] = nms(boxes, scores, iouThr, varargin)
% Greedy Non-Max Suppression on [x y w h] boxes by descending score.

if nargin<3 || isempty(iouThr), iouThr = 0.3; end  % IoU threshold (changeable)

if isempty(boxes)
    keepB = zeros(0,4); keepS = zeros(0,1); keepIdx = [];
    return;
end

% sort by score (desc) and iterate with removal to avoid infinite loop when no boxes
[~, order] = sort(scores(:), 'descend');
order = order(:);  % ensure column

keepIdx = zeros(numel(order), 1);  % preallocate max
nKeep = 0;
while ~isempty(order)
    i = order(1);                    % take best remaining
    nKeep = nKeep + 1;
    keepIdx(nKeep) = i;

    if numel(order) == 1
        break;
    end

    % IoU w.r.t. remaining boxes
    rem = order(2:end);
    xx1 = max(boxes(i,1), boxes(rem,1));
    yy1 = max(boxes(i,2), boxes(rem,2));
    xx2 = min(boxes(i,1)+boxes(i,3)-1, boxes(rem,1)+boxes(rem,3)-1);
    yy2 = min(boxes(i,2)+boxes(i,4)-1, boxes(rem,2)+boxes(rem,4)-1);

    w = max(0, xx2 - xx1 + 1);
    h = max(0, yy2 - yy1 + 1);
    inter = w .* h;

    area1 = boxes(i,3) * boxes(i,4);
    area2 = boxes(rem,3) .* boxes(rem,4);
    iou = inter ./ (area1 + area2 - inter);

    % keep boxes whose IoU is below threshold
    order = order([true; iou <= iouThr]);            % always drop the current best
end

keepIdx = keepIdx(1:nKeep);                          % already in descending-score order
keepB = boxes(keepIdx, :);
keepS = scores(keepIdx);
end
