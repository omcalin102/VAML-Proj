function I = iou(A, B, varargin)
% IoU between [x y w h] boxes; A: Px4, B: Gx4 → I: PxG

p = inputParser;
addParameter(p, 'Epsilon', 1e-12);      % numeric epsilon (changeable)
parse(p, varargin{:});
epsi = p.Results.Epsilon;

if isempty(A) || isempty(B)
    I = zeros(size(A,1), size(B,1));
    return;
end

% to [x1 y1 x2 y2]
Axy1 = [A(:,1), A(:,2)];
Axy2 = [A(:,1)+A(:,3)-1, A(:,2)+A(:,4)-1];
Bxy1 = [B(:,1), B(:,2)];
Bxy2 = [B(:,1)+B(:,3)-1, B(:,2)+B(:,4)-1];

P = size(A,1); G = size(B,1);
I = zeros(P,G);

for p = 1:P
    ax1 = Axy1(p,1); ay1 = Axy1(p,2); ax2 = Axy2(p,1); ay2 = Axy2(p,2);
    areaA = max(0, ax2-ax1+1) * max(0, ay2-ay1+1);
    for g = 1:G
        bx1 = Bxy1(g,1); by1 = Bxy1(g,2); bx2 = Bxy2(g,1); by2 = Bxy2(g,2);
        areaB = max(0, bx2-bx1+1) * max(0, by2-by1+1);

        ix1 = max(ax1,bx1); iy1 = max(ay1,by1);
        ix2 = min(ax2,bx2); iy2 = min(ay2,by2);
        iw  = max(0, ix2-ix1+1);
        ih  = max(0, iy2-iy1+1);
        inter = iw*ih;

        I(p,g) = inter / max(epsi, areaA + areaB - inter);  % epsilon (changeable)
    end
end
end
