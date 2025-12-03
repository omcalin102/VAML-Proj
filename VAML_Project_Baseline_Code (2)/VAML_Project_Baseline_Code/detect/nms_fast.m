function [B, S] = nms_fast(B, S, iouThr)
% Simple NMS: keep high scores; suppress boxes with IoU >= iouThr.
if isempty(B), return; end
S = S(:); [S, idx] = sort(S, 'descend'); B = B(idx,:);
x1=B(:,1); y1=B(:,2); x2=x1+B(:,3)-1; y2=y1+B(:,4)-1;
areas=(x2-x1+1).*(y2-y1+1);
keep = true(size(S));
for i=1:numel(S)
    if ~keep(i), continue; end
    xx1=max(x1(i),x1); yy1=max(y1(i),y1);
    xx2=min(x2(i),x2); yy2=min(y2(i),y2);
    w=max(0,xx2-xx1+1); h=max(0,yy2-yy1+1);
    inter=w.*h;
    iou = inter ./ max(1e-9, areas(i)+areas-inter);
    sup = (iou>=iouThr); sup(1:i)=false; keep(sup)=false;
end
B = B(keep,:); S = S(keep,:);
end
