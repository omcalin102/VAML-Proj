function [tp, fp, fn, matches] = match_detections(predBoxes, gtBoxes, iouThr)
% Returns TP/FP/FN and [idxPred, idxGT, IoU] matches

if nargin < 3, iouThr = 0.5; end                  % <-- change IoU threshold here

tp = 0; fp = 0; fn = 0; matches = zeros(0,3);

P = size(predBoxes,1); G = size(gtBoxes,1);
if P==0 && G==0, return; end
if P==0, fn = G; return; end
if G==0, fp = P; return; end

% Compute IoU matrix
IoU = bbox_iou_matrix(predBoxes, gtBoxes);

% Greedy match: repeatedly pick max IoU >= thr
usedP = false(P,1); usedG = false(G,1);
while true
    [mx, idx] = max(IoU(:));
    if isempty(mx) || mx < iouThr, break; end     % <-- IoU threshold
    [ip, ig] = ind2sub(size(IoU), idx);
    if ~usedP(ip) && ~usedG(ig)
        usedP(ip) = true; usedG(ig) = true;
        tp = tp + 1;
        matches(end+1,:) = [ip, ig, mx]; %#ok<AGROW>
    end
    IoU(ip,:) = -Inf; IoU(:,ig) = -Inf;           % block row/col
end

fp = sum(~usedP);
fn = sum(~usedG);
end

% -------- helpers --------
function I = bbox_iou_matrix(A, B)
% A: Px4 [x y w h], B: Gx4
if isempty(A) || isempty(B), I = zeros(size(A,1), size(B,1)); return; end
Axy1 = [A(:,1), A(:,2)];
Axy2 = [A(:,1)+A(:,3)-1, A(:,2)+A(:,4)-1];
Bxy1 = [B(:,1), B(:,2)];
Bxy2 = [B(:,1)+B(:,3)-1, B(:,2)+B(:,4)-1];

P = size(A,1); G = size(B,1);
I = zeros(P,G);
for p = 1:P
    ax1 = Axy1(p,1); ay1 = Axy1(p,2); ax2 = Axy2(p,1); ay2 = Axy2(p,2);
    for g = 1:G
        bx1 = Bxy1(g,1); by1 = Bxy1(g,2); bx2 = Bxy2(g,1); by2 = Bxy2(g,2);
        ix1 = max(ax1,bx1); iy1 = max(ay1,by1);
        ix2 = min(ax2,bx2); iy2 = min(ay2,by2);
        iw = max(0, ix2 - ix1 + 1);
        ih = max(0, iy2 - iy1 + 1);
        inter = iw * ih;
        if inter == 0, continue; end
        areaA = (ax2-ax1+1)*(ay2-ay1+1);
        areaB = (bx2-bx1+1)*(by2-by1+1);
        I(p,g) = inter / (areaA + areaB - inter);
    end
end
end
