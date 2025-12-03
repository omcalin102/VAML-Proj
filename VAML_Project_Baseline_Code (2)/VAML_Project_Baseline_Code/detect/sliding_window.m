function wins = sliding_window(I, winSize, step)
% Return [x y w h] windows over image I (grayscale or RGB).

if size(I,3)>1, [H,W,~]=size(I); else, [H,W]=size(I); end
w = winSize(1); h = winSize(2);
xs = 1:step:max(1, W - w + 1);          % stride in x (changeable via 'step')
ys = 1:step:max(1, H - h + 1);          % stride in y (changeable via 'step')

n = numel(xs)*numel(ys);
wins = zeros(n,4,'uint32');
k=0;
for yy = ys
    for xx = xs
        k=k+1;
        wins(k,:) = [xx, yy, w, h];
    end
end
end
