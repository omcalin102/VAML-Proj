function demo_detection_sweep()
% Evaluate multiple detector hyper-parameter versions with metrics + timings.
% Produces CSV/figure artifacts to justify operating-point choices.

clc; close all; rng(42,'twister'); addpath(genpath('.'));

% ---- PATHS ----
frameDir    = fullfile('pedestrian','pedestrian');
gtFile      = fullfile('data','test.dataset');
modelPath   = fullfile('results','models','model_baseline.mat');
outTableDir = fullfile('results','tables');
outFigureDir= fullfile('report','figs');
ensure_dir(outTableDir, outFigureDir);

% ---- DETECTOR VERSIONS (3–6 configs) ----
configs = {
    struct('Name','fast','Step',12,'Scale',0.92,'NMS',0.30,'MinScore',-Inf), ...
    struct('Name','balanced','Step',8,'Scale',0.90,'NMS',0.35,'MinScore',0.0), ...
    struct('Name','accurate','Step',6,'Scale',0.88,'NMS',0.40,'MinScore',0.5), ...
    struct('Name','tightNMS','Step',8,'Scale',0.90,'NMS',0.50,'MinScore',0.0), ...
    struct('Name','highThresh','Step',8,'Scale',0.90,'NMS',0.35,'MinScore',1.0)
};

% ---- LOAD MODEL / GT ----
assert(exist(modelPath,'file')==2, 'Model not found: %s', modelPath);
S = load(modelPath); model = S.model;
assert(exist(gtFile,'file')==2 && exist('load_gt','file')==2, 'Ground truth utilities missing.');
GT = load_gt(gtFile);
frames = dir(fullfile(frameDir,'*.jpg')); if isempty(frames), frames = dir(fullfile(frameDir,'*.png')); end
assert(~isempty(frames), 'No frames found in %s', frameDir);

rows = {};

for c = 1:numel(configs)
    cfg = configs{c};
    fprintf('[%d/%d] %s ...\n', c, numel(configs), cfg.Name);
    loopTimer = tic;
    allTP=0; allFP=0; allFN=0; perFrameMs = zeros(numel(frames),1);
    for k = 1:numel(frames)
        I = imread(fullfile(frames(k).folder, frames(k).name));
        t0 = tic;
        [boxes, scores] = score_windows(I, model, ...
            'BaseWindow',[64 128], 'Step',cfg.Step, 'ScaleFactor',cfg.Scale, 'MinScore',cfg.MinScore);
        [keepB, ~] = nms(boxes, scores, cfg.NMS);
        perFrameMs(k) = toc(t0)*1000;

        key = strip_extension(frames(k).name);
        if isfield(GT, key) && ~isempty(GT.(key))
            [tp,fp,fn] = match_detections(keepB, GT.(key), 0.5);
            allTP=allTP+tp; allFP=allFP+fp; allFN=allFN+fn;
        end

        if mod(k, max(1,floor(numel(frames)/10)))==0 || k==numel(frames)
            progress_bar(k, numel(frames), loopTimer, '  - frames');
        end
    end

    prec = allTP / max(1, allTP+allFP);
    rec  = allTP / max(1, allTP+allFN);
    f1   = 2*prec*rec / max(1e-9, prec+rec);
    rows(end+1,:) = {cfg.Name, cfg.Step, cfg.Scale, cfg.NMS, cfg.MinScore, prec, rec, f1, mean(perFrameMs)}; %#ok<AGROW>
    fprintf('  P=%.3f R=%.3f F1=%.3f | %.1f ms/frame\n', prec, rec, f1, mean(perFrameMs));
end

T = cell2table(rows, 'VariableNames',{'Name','Step','ScaleFactor','NMS_IoU','MinScore','Precision','Recall','F1','MsPerFrame'});
csvPath = fullfile(outTableDir, 'detection_sweep.csv');
writetable(T, csvPath);
fprintf('Saved sweep table: %s\n', csvPath);

fig = figure('Color','w','Position',[100 100 760 360]);
bar(categorical(T.Name), T.F1); grid on; ylim([0 1]); ylabel('F1');
title('Detector versions (F1)');
figPath = fullfile(outFigureDir, 'detection_sweep_f1.png');
exportgraphics(fig, figPath, 'Resolution', 150); close(fig);
fprintf('Saved figure: %s\n', figPath);

fprintf('DONE.\n');
end

function ensure_dir(varargin)
for i = 1:nargin
    d = varargin{i};
    if exist(d,'dir'), continue; end
    [status,msg] = mkdir(d); % mkdir creates intermediate folders when needed
    if ~status
        error('Failed to create directory %s: %s', d, msg);
    end
end
end
function s = strip_extension(fname)
[~,s,~] = fileparts(fname);
end
