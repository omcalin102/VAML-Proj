function [boxes, scores] = score_windows(I, model, varargin)
% Slide a window (multi-scale) and score with a trained SVM (fast, capped).

p = inputParser;
addParameter(p,'BaseWindow',[64 128]);   % [w h] of detector window
addParameter(p,'Step',16);               % stride in pixels
addParameter(p,'ScaleFactor',0.90);      % pyramid factor (<1 shrinks per level)
addParameter(p,'MinScore',-Inf);         % keep boxes with score >= MinScore
addParameter(p,'MaxWindows',200);        % hard cap for speed
addParameter(p,'Verbose',true);          % print progress
parse(p,varargin{:}); a = p.Results;

% grayscale uint8 (HOG expects single-channel)
if size(I,3)>1, I = rgb2gray(I); end
if ~isa(I,'uint8'), I = im2uint8(I); end

% 1) multi-scale windows
scales = pyramid_scales(size(I), a.BaseWindow, a.ScaleFactor);
winScaled = cell(numel(scales),1);
winOrig   = cell(numel(scales),1);
winScale  = cell(numel(scales),1);
for s = 1:numel(scales)
    Is = imresize(I, scales{s});
    ws = sliding_window(Is, a.BaseWindow, a.Step);
    winScaled{s} = ws;
    winOrig{s}   = [ws(:,1:2)./scales{s}, ws(:,3:4)./scales{s}];
    winScale{s}  = repmat(scales{s}, size(ws,1), 1);
end
winsAll = vertcat(winOrig{:});
winsScaledAll = vertcat(winScaled{:});
scaleVec = vertcat(winScale{:});
nW = size(winsAll,1);
if nW==0, boxes=zeros(0,4,'uint32'); scores=zeros(0,1); return; end
if nW > a.MaxWindows
    idx = round(linspace(1, nW, a.MaxWindows))';
    winsAll = winsAll(idx,:);
    winsScaledAll = winsScaledAll(idx,:);
    scaleVec = scaleVec(idx);
    nW = size(winsAll,1);
end
if a.Verbose, fprintf('  - windows: %d (capped) across %d scales\n', nW, numel(scales)); drawnow; end

% 2) HOG features (per scale to avoid repeated resizes)
tH = tic;
feats = [];
uniqueScales = unique(scaleVec, 'stable');
for u = 1:numel(uniqueScales)
    sc = uniqueScales(u);
    Is = imresize(I, sc);
    idx = find(scaleVec==sc);
    for ii = 1:numel(idx)
        k = idx(ii);
        f = extract_hog(imcrop_safe(Is, winsScaledAll(k,:)), 'ResizeTo', a.BaseWindow);
        if isempty(feats)
            D = numel(f); feats = zeros(nW, D, 'single');
        end
        feats(k,:) = f;
        if a.Verbose && mod(k,50)==0
            fprintf('  - HOG %4d/%4d (%.2fs)\n', k, nW, toc(tH)); drawnow;
        end
    end
end
if a.Verbose, fprintf('  - HOG total: %.2fs\n', toc(tH)); drawnow; end

% 3) Predict → use positive-class margin
tP = tic;
[~, sc] = predict(model, feats);
if size(sc,2) >= 2
    cls = string(model.ClassNames);
    posIx = find(cls=="1" | lower(cls)=="pos" | cls=="true", 1);  % ← positive class
    if isempty(posIx), posIx = size(sc,2); end                    % fallback
    negIx = setdiff(1:size(sc,2), posIx);
    sc = sc(:,posIx) - max(sc(:,negIx),[],2);                      % margin = pos - best other
else
    sc = sc(:,1);
end
if a.Verbose, fprintf('  - predict: %d×%d -> %.2fs\n', size(feats,1), size(feats,2), toc(tP)); drawnow; end

% 4) Prefilter
keep  = sc >= a.MinScore;                 % ← raise to be stricter
boxes = winsAll(keep,:);
scores= sc(keep);
if a.Verbose, fprintf('  - prefilter kept: %d\n', numel(scores)); drawnow; end
end

% ------- helpers -------
function wins = sliding_window(I, wh, step)
W = size(I,2); H = size(I,1); w = wh(1); h = wh(2);
xs = 1:step:max(1, W-w+1); ys = 1:step:max(1, H-h+1);
[XX,YY] = meshgrid(xs,ys);
wins = [XX(:) YY(:) repmat([w h], numel(XX), 1)];
end

function scales = pyramid_scales(imgSize, baseWindow, scaleFactor)
H = imgSize(1); W = imgSize(2); h = baseWindow(2); w = baseWindow(1);
scales = {};
current = 1;
while H*current >= h && W*current >= w
    scales{end+1,1} = current; %#ok<AGROW>
    if scaleFactor >= 0.999, break; end
    current = current * scaleFactor;
end
end

function P = imcrop_safe(I, box)
x = max(1, box(1)); y = max(1, box(2)); w = max(1, box(3)); h = max(1, box(4));
x2 = min(size(I,2), x+w-1); y2 = min(size(I,1), y+h-1);
P = I(y:y2, x:x2);
end
