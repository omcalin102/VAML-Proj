function eval_metrics_imageset(varargin)
% Evaluate detector on images; compute TP/FP/FN, Precision/Recall/F1, PR curve, AP.
% Writes into <OutDirBase>/<RunTag>/
%
% Key additions:
% - Robust GT lookup (handles zero-padding / extensions)
% - Audit CSV of per-image matches
% - PR curve CSV + PNG
% - AP (area under PR by trapezoid)
% - Operating point selection: by F1 or by precision floor

p = inputParser;
addParameter(p,'ImageDir','');
addParameter(p,'ModelPath',fullfile('results','models','model_baseline.mat'));
addParameter(p,'BaseWindow',[64 128]);
addParameter(p,'Step',16);
addParameter(p,'ResizeLongSide',480);
addParameter(p,'MinScore',0.6);
addParameter(p,'NMS_IoU',0.25);
addParameter(p,'GTFile','');
addParameter(p,'IoU_TP',0.50);
addParameter(p,'RunTag','run5');
addParameter(p,'OutDirBase','');     % if empty, infer from ModelPath
addParameter(p,'MakePR',true);
addParameter(p,'SelectOp','byF1');   % 'byF1' | 'byPrecisionFloor'
addParameter(p,'PrecisionFloor',0.85);
parse(p,varargin{:}); a = p.Results;

assert(~isempty(a.ImageDir) && exist(a.ImageDir,'dir')==7, 'Bad ImageDir: %s', a.ImageDir);
assert(exist(a.ModelPath,'file')==2, 'Model not found: %s', a.ModelPath);

if isempty(a.OutDirBase)
    projRoot = fileparts(fileparts(a.ModelPath));
    a.OutDirBase = fullfile(projRoot, 'results', 'metrics');
end
outDir = fullfile(a.OutDirBase, a.RunTag);
safe_mkdir(outDir);

S = load(a.ModelPath); model = S.model;

imgs = [dir(fullfile(a.ImageDir,'*.jpg')); dir(fullfile(a.ImageDir,'*.png')); ...
        dir(fullfile(a.ImageDir,'*.JPG')); dir(fullfile(a.ImageDir,'*.PNG'))];
assert(~isempty(imgs),'No images in %s', a.ImageDir);

% GT map
gtMap = containers.Map('KeyType','char','ValueType','any');
if ~isempty(a.GTFile) && exist(a.GTFile,'file')==2
    gtMap = parse_dataset_qub(a.GTFile);
    fprintf('Loaded GT from %s with %d keys.\n', a.GTFile, gtMap.Count);
end

rows = {};
TPtotal=0; FPtotal=0; FNtotal=0; t_ms = 0;

% also build an audit (image, FoundGT, UsedKey, NumGT)
audit = {};

fprintf('Evaluating %d image(s) → %s\n', numel(imgs), outDir);
for k=1:numel(imgs)
    name  = imgs(k).name; inpth = fullfile(imgs(k).folder, name);
    I0 = imread(inpth);
    [H,W,~] = size(I0); longSide = max(H,W);
    scale=1.0; I=I0; stepWork=a.Step;
    if a.ResizeLongSide>0 && longSide>a.ResizeLongSide
        scale = a.ResizeLongSide/double(longSide);
        I = imresize(I0, scale);
        stepWork = max(4, round(a.Step * scale));
    end

    t0=tic;
    [B,S] = score_windows(I,model,'BaseWindow',a.BaseWindow,'Step',stepWork, ...
                          'MinScore',a.MinScore,'MaxWindows',200,'Verbose',false);
    [B,S] = nms_fast(B,S,a.NMS_IoU);
    if scale~=1.0, B = round(B./scale); end
    ms = toc(t0)*1000; t_ms = t_ms + ms;

    [has, GT, usedKey] = gt_lookup(gtMap, name);
    if has
        [TP,FP,FN,prec,rec,f1] = pr_counts(B,GT,a.IoU_TP);
        TPtotal=TPtotal+TP; FPtotal=FPtotal+FP; FNtotal=FNtotal+FN;
        rows(end+1,:)  = {name, size(B,1), size(GT,1), TP, FP, FN, prec, rec, f1, ms}; %#ok<AGROW>
        audit(end+1,:) = {name, true, usedKey, size(GT,1)}; %#ok<AGROW>
    else
        rows(end+1,:)  = {name, size(B,1), NaN, NaN, NaN, NaN, NaN, NaN, NaN, ms}; %#ok<AGROW>
        audit(end+1,:) = {name, false, '', 0}; %#ok<AGROW>
    end
end

% Write per-image metrics & audit
T = cell2table(rows,'VariableNames', ...
    {'Image','NumDet','NumGT','TP','FP','FN','Precision','Recall','F1','MsPerImage'});
writetable(T, fullfile(outDir, 'image_metrics.csv'));

A = cell2table(audit,'VariableNames',{'Image','FoundGT','UsedGTKey','NumGT'});
writetable(A, fullfile(outDir, 'alignment_audit.csv'));

% Global summary
if TPtotal+FPtotal>0, Prec = TPtotal/(TPtotal+FPtotal); else, Prec = NaN; end
if TPtotal+FNtotal>0, Rec  = TPtotal/(TPtotal+FNtotal); else, Rec  = NaN; end
F1 = 2*Prec*Rec/(Prec+Rec);
fid=fopen(fullfile(outDir,'summary.txt'),'w');
fprintf(fid,'TP=%d FP=%d FN=%d | P=%.3f R=%.3f F1=%.3f | mean(ms/img)=%.1f\n', ...
    TPtotal,FPtotal,FNtotal,Prec,Rec,F1,t_ms/numel(imgs));
fclose(fid);

% PR sweep + AP + op-point selection
if a.MakePR && gtMap.Count>0
    ths = 0.40 : 0.05 : 0.80;  % focused sweep
    PR  = zeros(numel(ths), 3); % [P R F1]
    for i=1:numel(ths)
        a2=a; a2.MinScore=ths(i);
        [P_i, R_i, F1_i] = sweep_once(imgs, a2, model, gtMap);
        PR(i,:) = [P_i, R_i, F1_i];
    end
    % AP by trapezoid over recall (sorted by recall)
    [~,ix] = sort(PR(:,2)); PRs = PR(ix,:); ths_s = ths(ix);
    AP = trapz(PRs(:,2), PRs(:,1));

    % choose operating point
    switch lower(a.SelectOp)
        case 'byprecisionfloor'
            mask = PR(:,1) >= a.PrecisionFloor;
            if any(mask)
                [~, j] = max(PR(mask,3));    % best F1 under precision floor
                idx = find(mask); j = idx(j);
            else
                [~, j] = max(PR(:,3));       % fallback: max F1
            end
        otherwise % byF1
            [~, j] = max(PR(:,3));
    end
    op = struct('Threshold', ths(j), 'P', PR(j,1), 'R', PR(j,2), 'F1', PR(j,3), ...
                'AP', AP, 'PrecisionFloor', a.PrecisionFloor, 'Policy', a.SelectOp);

    % write artifacts
    csv = table(ths(:), PR(:,1), PR(:,2), PR(:,3), 'VariableNames', {'Threshold','Precision','Recall','F1'});
    writetable(csv, fullfile(outDir,'pr_curve.csv'));
    save(fullfile(outDir,'pr_curve.mat'),'PR','ths','AP','op');

    f = figure('Visible','off');
    plot(PR(:,2), PR(:,1), '-o'); grid on; xlabel('Recall'); ylabel('Precision'); title('PR (threshold sweep)');
    hold on; plot(op.R, op.P, 'r*', 'MarkerSize', 10); hold off;
    saveas(f, fullfile(outDir,'pr_curve.png'));

    fid=fopen(fullfile(outDir,'operating_point.txt'),'w');
    fprintf(fid,'Policy=%s PrecisionFloor=%.2f\n', a.SelectOp, a.PrecisionFloor);
    fprintf(fid,'Chosen: Threshold=%.3f | P=%.3f R=%.3f F1=%.3f | AP=%.3f\n', ...
        op.Threshold, op.P, op.R, op.F1, op.AP);
    fclose(fid);
end

fprintf('Saved metrics to %s\n', outDir);
end

% ------- helpers -------
function safe_mkdir(d); if exist(d,'dir')~=7, assert(mkdir(d), 'mkdir failed: %s', d); end; end

function [P,R,F1] = sweep_once(imgs, a, model, gtMap)
TP=0; FP=0; FN=0;
for k=1:numel(imgs)
    name=imgs(k).name; inpth=fullfile(imgs(k).folder,name);
    I0=imread(inpth);
    [H,W,~]=size(I0); longSide=max(H,W);
    scale=1.0; I=I0; stepWork=a.Step;
    if a.ResizeLongSide>0 && longSide>a.ResizeLongSide
        scale=a.ResizeLongSide/double(longSide);
        I=imresize(I0,scale); stepWork=max(4, round(a.Step*scale));
    end
    [B,S]=score_windows(I,model,'BaseWindow',a.BaseWindow,'Step',stepWork, ...
                        'MinScore',a.MinScore,'MaxWindows',200,'Verbose',false);
    [B,S]=nms_fast(B,S,a.NMS_IoU);
    if scale~=1.0, B=round(B./scale); end

    [has, GT] = gt_lookup(gtMap, name);
    if has
        [tp,fp,fn]=pr_counts(B,GT,a.IoU_TP); TP=TP+tp; FP=FP+fp; FN=FN+fn;
    end
end
P = TP/(TP+FP+eps); R = TP/(TP+FN+eps); F1 = 2*P*R/(P+R+eps);
end

function [TP,FP,FN,prec,rec,f1] = pr_counts(B,GT,thr)
m=size(B,1); n=size(GT,1);
TP=0; FP=0; FN=0; used=false(n,1);
for i=1:m
    [mx,ix]=max(iou(B(i,:),GT));
    if ~isempty(mx) && mx>=thr && ~used(ix)
        TP=TP+1; used(ix)=true;
    else
        FP=FP+1;
    end
end
FN = sum(~used);
prec = TP/(TP+FP+eps);
rec  = TP/(TP+FN+eps);
f1   = 2*prec*rec/(prec+rec+eps);
end

function v = iou(b,B)
if isempty(B), v=[]; return; end
x1=B(:,1); y1=B(:,2); x2=x1+B(:,3)-1; y2=y1+B(:,4)-1;
xx1=max(b(1),x1); yy1=max(b(2),y1);
xx2=min(b(1)+b(3)-1,x2); yy2=min(b(2)+b(4)-1,y2);
w=max(0,xx2-xx1+1); h=max(0,yy2-yy1+1);
inter=w.*h; area1=b(3)*b(4); area2=(x2-x1+1).*(y2-y1+1);
v = inter./max(1e-9, area1+area2-inter);
end
