function [boxes, scores] = score_windows(I, model, varargin)
% Slide a window across a multi-scale pyramid and score with a trained model.

p = inputParser;
addParameter(p,'BaseWindow',[64 128]);   % [w h] of detector window
addParameter(p,'Step',16);               % stride in pixels
addParameter(p,'ScaleFactor',0.90);      % pyramid factor (0.85–0.95 typical)
addParameter(p,'MinScore',-Inf);         % keep boxes with score >= MinScore
addParameter(p,'MaxWindows',400);        % hard cap for speed
addParameter(p,'Verbose',true);          % print progress
addParameter(p,'Descriptor',[]);         % optional override descCfg
parse(p,varargin{:}); a = p.Results;

% Determine descriptor (fall back to HOG defaults)
if isstruct(model) && isfield(model,'Descriptor') && ~isempty(model.Descriptor)
    descCfg = model.Descriptor;
elseif ~isempty(a.Descriptor)
    descCfg = a.Descriptor;
else
    descCfg = struct('Type','hog','ResizeTo',a.BaseWindow,'CellSize',[8 8], ...
        'BlockSize',[2 2],'BlockOverlap',[1 1],'NumBins',9,'PCA',[]);
end
if ~isfield(descCfg,'ResizeTo') || isempty(descCfg.ResizeTo)
    descCfg.ResizeTo = a.BaseWindow;
end

% grayscale uint8
if size(I,3)>1, I = rgb2gray(I); end
if ~isa(I,'uint8'), I = im2uint8(I); end

% Build pyramid scales
scales = pyramid_scales(size(I), descCfg.ResizeTo, a.ScaleFactor, 'MinShortSide', min(descCfg.ResizeTo));

% Preallocate containers
boxes = zeros(a.MaxWindows, 4, 'single');
scores = zeros(a.MaxWindows, 1, 'single');
featDim = [];
nKept = 0;
maxPerScale = ceil(a.MaxWindows / numel(scales));   % spread windows across scales

tFeat = tic;
for sIdx = 1:numel(scales)
    remaining = a.MaxWindows - nKept;
    if remaining <= 0, break; end
    scale = scales(sIdx);
    I_s = imresize(I, scale);
    wins = sliding_window(I_s, descCfg.ResizeTo, a.Step);
    if isempty(wins), continue; end

    % Cap samples per scale to avoid exhausting the window budget on the first level
    scaleBudget = min(remaining, maxPerScale);
    if size(wins,1) > scaleBudget
        wins = wins(round(linspace(1, size(wins,1), scaleBudget)),:);
    end

    for k = 1:size(wins,1)
        if nKept >= a.MaxWindows, break; end
        nKept = nKept + 1;
        patch = imcrop_safe(I_s, wins(k,:));
        f = extract_descriptor(patch, descCfg);
        if isempty(featDim)
            featDim = numel(f);
            feats = zeros(a.MaxWindows, featDim, 'single');
        end
        feats(nKept,:) = f;
        boxes(nKept,:) = single([wins(k,1:2) ./ scale, wins(k,3:4) ./ scale]);
    end
end

if isempty(featDim)
    boxes = zeros(0,4,'single'); scores = zeros(0,1,'single'); return; end

feats = feats(1:nKept,:); boxes = boxes(1:nKept,:);
if a.Verbose
    fprintf('  - pyramid levels: %d | windows used: %d | feat %.2fs\n', numel(scales), nKept, toc(tFeat)); drawnow;
end

% 3) Predict → use positive-class margin
tP = tic;
[~, sc] = predict(get_classifier(model), feats);
if size(sc,2) >= 2
    cls = string(get_classifier(model).ClassNames);
    posIx = find(cls=="1" | lower(cls)=="pos" | cls=="true", 1);  % ← positive class
    if isempty(posIx), posIx = size(sc,2); end                      % fallback
    negIx = setdiff(1:size(sc,2), posIx);
    sc = sc(:,posIx) - max(sc(:,negIx),[],2);                        % margin = pos - best other
else
    sc = sc(:,1);
end
if a.Verbose, fprintf('  - predict: %d×%d -> %.2fs\n', size(feats,1), size(feats,2), toc(tP)); drawnow; end

% 4) Prefilter
keep  = sc >= a.MinScore;                 % ← raise to be stricter
boxes = boxes(keep,:);
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

function P = imcrop_safe(I, box)
x = max(1, box(1)); y = max(1, box(2)); w = max(1, box(3)); h = max(1, box(4));
x2 = min(size(I,2), x+w-1); y2 = min(size(I,1), y+h-1);
P = I(y:y2, x:x2);
end

function clf = get_classifier(model)
if isstruct(model) && isfield(model,'Classifier')
    clf = model.Classifier;
else
    clf = model;
end
end
