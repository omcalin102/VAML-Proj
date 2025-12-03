function eval_detection_set(varargin)
% Runs detector on a set of frames, saves per-frame PNGs with boxes,
% and writes rich metrics CSVs (per-frame, summary, PR sweep).

% ---- Tunables ----
p = inputParser;
addParameter(p,'ModelPath',fullfile('results','models','model_baseline.mat'));    % changeable
addParameter(p,'FrameDir',fullfile('pedestrian','pedestrian'));                   % changeable
addParameter(p,'GTFile',fullfile('data','test.dataset'));                         % changeable
addParameter(p,'OutFramesDir',fullfile('results','frames'));                      % changeable
addParameter(p,'OutTablesDir',fullfile('results','tables'));                      % changeable
addParameter(p,'BaseWindow',[64 128]);        % changeable
addParameter(p,'Step',8);                     % changeable
addParameter(p,'ScaleFactor',0.90);           % changeable
addParameter(p,'NMS_IoU',0.30);               % changeable
addParameter(p,'IoU_TP',0.50);                % changeable
addParameter(p,'MinScore',-Inf);              % changeable
addParameter(p,'MaxFrames',Inf);              % changeable
addParameter(p,'ScoreSweep',linspace(-2,3,11)); % thresholds for PR sweep (changeable)
parse(p,varargin{:});
args = p.Results;

% ---- Setup ----
assert(exist(args.ModelPath,'file')==2, 'Model not found: %s', args.ModelPath);
S = load(args.ModelPath); model = S.model;

if ~exist(args.OutFramesDir,'dir'), mkdir(args.OutFramesDir); end
if ~exist(args.OutTablesDir,'dir'), mkdir(args.OutTablesDir); end

frames = dir(fullfile(args.FrameDir,'*.jpg'));
if isempty(frames), frames = dir(fullfile(args.FrameDir,'*.png')); end
assert(~isempty(frames),'No frames in %s', args.FrameDir);
if isfinite(args.MaxFrames), frames = frames(1:min(args.MaxFrames,numel(frames))); end

% Ground truth (optional but recommended)
haveGT = false; GT = [];
if exist(args.GTFile,'file')==2 && exist('load_gt','file')==2
    try, GT = load_gt(args.GTFile); haveGT = true; catch, haveGT = false; end
end

% ---- Aggregators ----
rows = [];   % per-frame metrics
allDet = []; % [frameIdx, score, x y w h]
allGT  = []; % [frameIdx, x y w h]
allIoUTP = []; % IoU of TPs across all frames
perFrameMs = zeros(numel(frames),1);

% ---- Loop frames ----
for k=1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));
    t0 = tic;
    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',args.BaseWindow, ...     % changeable
        'Step',args.Step, ...                 % changeable
        'ScaleFactor',args.ScaleFactor, ...   % changeable
        'MinScore',args.MinScore);            % changeable
    [keepB, keepS] = nms(boxes, scores, args.NMS_IoU);  % changeable
    perFrameMs(k) = toc(t0)*1000;

    % Save raw detections for PR sweep/AP
    allDet = [allDet; [k*ones(size(keepB,1),1), keepS, keepB]]; %#ok<AGROW>

    % Load GT for this frame (if available)
    key = strip_ext(frames(k).name);
    gtB = [];
    if haveGT && isfield(GT,key), gtB = GT.(key); end
    if ~isempty(gtB)
        allGT = [allGT; [k*ones(size(gtB,1),1), gtB]]; %#ok<AGROW>
        [tp,fp,fn, matches] = match_detections(keepB, gtB, args.IoU_TP); % changeable
        % IoU of TPs (if requested)
        for m=1:size(matches,1)
            ip = matches(m,1); ig = matches(m,2);
            iouVal = iou(keepB(ip,:), gtB(ig,:));
            allIoUTP(end+1,1) = iouVal; %#ok<AGROW>
        end
    else
        tp=0; fp=size(keepB,1); fn=0; matches = zeros(0,3);
    end

    % Per-frame metrics
    prec = tp / max(1, tp+fp);
    rec  = tp / max(1, tp+fn);
    f1   = 2*prec*rec / max(1e-9, prec+rec);
    rows = [rows; {key, size(gtB,1), size(keepB,1), tp, fp, fn, prec, rec, f1, perFrameMs(k)}]; %#ok<AGROW>

    % ---- Write annotated frame ----
    J = I;
    if ~isempty(keepB)
        J = draw_boxes(J, keepB, keepS, 'Color','red');     % changeable
    end
    if ~isempty(gtB)
        J = insertShape(J,'Rectangle',gtB,'Color','green','LineWidth',2); % changeable
    end
    imwrite(J, fullfile(args.OutFramesDir, sprintf('%s_det.png', key)));
end

% ---- Write per-frame table ----
T = cell2table(rows, 'VariableNames', ...
    {'Frame','NumGT','NumDet','TP','FP','FN','Precision','Recall','F1','MsPerFrame'});
writetable(T, fullfile(args.OutTablesDir,'detection_per_frame.csv'));

% ---- Summary metrics ----
sumTP = sum(T.TP); sumFP = sum(T.FP); sumFN = sum(T.FN);
Prec = sumTP / max(1, sumTP + sumFP);
Rec  = sumTP / max(1, sumTP + sumFN);
F1   = 2*Prec*Rec / max(1e-9, Prec+Rec);
AvgIoU_TP = mean(allIoUTP(~isnan(allIoUTP))) * ( ~isempty(allIoUTP) ); % 0 if empty
AvgFPperImg = sumFP / height(T);
AvgDetPerImg = mean(T.NumDet);
MeanMs = mean(T.MsPerFrame);

% ---- PR Sweep + AP (approx) ----
[prTable, AP] = pr_sweep(allDet, allGT, args.ScoreSweep, args.IoU_TP); % changeable
writetable(prTable, fullfile(args.OutTablesDir,'pr_sweep.csv'));

S = table(Prec, Rec, F1, AvgIoU_TP, AvgFPperImg, AvgDetPerImg, MeanMs, AP, ...
    'VariableNames',{'Precision','Recall','F1','AvgIoU_TP','AvgFPperImg','AvgDetPerImg','MsPerFrame','AP'});
writetable(S, fullfile(args.OutTablesDir,'detection_summary.csv'));

fprintf('Saved per-frame PNGs to %s\n', args.OutFramesDir);
fprintf('Wrote tables: detection_per_frame.csv, detection_summary.csv, pr_sweep.csv\n');
end

% --- helpers ---
function s = strip_ext(f), [~,s,~] = fileparts(f); end

function val = iou(A,B)
% A,B: [x y w h]
ax1=A(1); ay1=A(2); ax2=A(1)+A(3)-1; ay2=A(2)+A(4)-1;
bx1=B(1); by1=B(2); bx2=B(1)+B(3)-1; by2=B(2)+B(4)-1;
ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2);
iw=max(0,ix2-ix1+1); ih=max(0,iy2-iy1+1);
inter=iw*ih; areaA=(ax2-ax1+1)*(ay2-ay1+1); areaB=(bx2-bx1+1)*(by2-by1+1);
val = inter / max(1e-12, areaA+areaB-inter);
end

function [T, AP] = pr_sweep(allDet, allGT, threshVec, iouThr)
% allDet: [frame, score, x y w h]; allGT: [frame, x y w h]
if nargin<4, iouThr=0.5; end
if isempty(allGT)
    T = table(threshVec(:), NaN(numel(threshVec),1), NaN(numel(threshVec),1), NaN(numel(threshVec),1), ...
        'VariableNames',{'ScoreThr','Precision','Recall','F1'});
    AP = NaN; return;
end
prec=zeros(numel(threshVec),1); rec=prec; f1=prec;
for i=1:numel(threshVec)
    thr = threshVec(i);
    D = allDet(allDet(:,2)>=thr, :);
    [tp,fp,fn] = eval_with_matching(D, allGT, iouThr);
    prec(i) = tp / max(1, tp+fp);
    rec(i)  = tp / max(1, tp+fn);
    f1(i)   = 2*prec(i)*rec(i) / max(1e-9, prec(i)+rec(i));
end
T = table(threshVec(:), prec, rec, f1, 'VariableNames',{'ScoreThr','Precision','Recall','F1'});
% AP (approx): area under PR via trapezoid (monotone fix optional)
AP = trapz(rec, prec);  % quick approximation
end

function [tp,fp,fn] = eval_with_matching(D, G, iouThr)
% D: [frame, score, x y w h]; G: [frame, x y w h]
tp=0; fp=0; fn=0;
frames = unique([D(:,1); G(:,1)]);
for f = frames'
    Df = D(D(:,1)==f, 3:6);
    Gf = G(G(:,1)==f, 2:5);
    if isempty(Df) && isempty(Gf), continue; end
    if isempty(Df), fn = fn + size(Gf,1); continue; end
    if isempty(Gf), fp = fp + size(Df,1); continue; end
    % greedy match by IoU
    P=size(Df,1); Gn=size(Gf,1);
    usedP=false(P,1); usedG=false(Gn,1);
    IoU = zeros(P,Gn);
    for i=1:P, for j=1:Gn, IoU(i,j)=iou(Df(i,:),Gf(j,:)); end, end
    while true
        [mx, idx] = max(IoU(:));
        if isempty(mx) || mx < iouThr, break; end
        [ip, ig] = ind2sub(size(IoU), idx);
        if ~usedP(ip) && ~usedG(ig)
            usedP(ip)=true; usedG(ig)=true; tp=tp+1;
        end
        IoU(ip,:)=-Inf; IoU(:,ig)=-Inf;
    end
    fp = fp + sum(~usedP);
    fn = fn + sum(~usedG);
end
end
