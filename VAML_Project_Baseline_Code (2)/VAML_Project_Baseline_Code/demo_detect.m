function demo_detect()
% Run detector with a saved model; saves video + optional detection metrics.

clc; close all; rng(42,'twister');                           % RNG seed (changeable)

% ---- PATHS ----
modelPath   = fullfile('results','models','model_best_combo.mat'); % model path (changeable)
frameDir    = fullfile('pedestrian','pedestrian');                % test frames dir (changeable)
gtFile      = fullfile('data','test.dataset');                    % GT file (optional) (changeable)
outVideoDir = fullfile('results','videos');                       % video out (changeable)
outTableDir = fullfile('results','tables');                       % tables out (changeable)
outFigureDir= fullfile('report','figs');                          % figs out (changeable)
ensure_dir(outVideoDir, outTableDir, outFigureDir);

% ---- DETECTOR HYPER-PARAMETERS ----
BaseWindow  = [64 128];        % window size [w h] (changeable, overridden by model descriptor)
Step        = 8;               % stride px (4/8/12) (changeable)
ScaleFactor = 0.90;            % pyramid factor (0.85–0.95) (changeable)
NMS_IoU     = 0.30;            % NMS IoU (0.3–0.5) (changeable)
MinScore    = -Inf;            % pre-NMS score filter (changeable)
MaxFrames   = 12;              % frames to process (changeable)
IoU_TP      = 0.50;            % TP IoU rule (changeable)
FPS         = 6;               % output video fps (changeable)

% ---- LOAD MODEL ----
assert(exist(modelPath,'file')==2, 'Model not found: %s', modelPath);
S = load(modelPath); model = S.model;
if isstruct(model) && isfield(model,'Descriptor')
    BaseWindow = model.Descriptor.ResizeTo;
end

% ---- COLLECT FRAMES ----
frames = dir(fullfile(frameDir,'*.jpg'));
if isempty(frames), frames = dir(fullfile(frameDir,'*.png')); end
assert(~isempty(frames), 'No frames found in %s', frameDir);
frames = frames(1:min(MaxFrames, numel(frames)));               % frame cap (changeable)

% ---- OPTIONAL: LOAD GT ----
haveGT=false; GT=[];
if exist(gtFile,'file')==2 && exist('load_gt','file')==2
    try, GT=load_gt(gtFile); haveGT=true; catch, haveGT=false; end
end

% ---- VIDEO WRITER ----
vidPath = fullfile(outVideoDir,'demo_detect.mp4');              % video name (changeable)
try
    vw = VideoWriter(vidPath, 'MPEG-4');                % MP4 (best if supported)
catch
    vidPath = fullfile(outVideoDir, 'demo_baseline.avi'); % fallback filename
    vw = VideoWriter(vidPath, 'Motion JPEG AVI');         % AVI fallback
end
vw.FrameRate = 6;  % changeable
open(vw);

% ---- RUN DETECTOR ----
allTP=0; allFP=0; allFN=0;
perFrameMs = zeros(numel(frames),1);
for k=1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));
    t0 = tic;
    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',BaseWindow, ...                           % window (changeable)
        'Step',Step, ...                                       % stride (changeable)
        'ScaleFactor',ScaleFactor, ...                         % pyramid factor (changeable)
        'MinScore',MinScore);                                  % pre-filter (changeable)
    [keepB, keepS] = nms(boxes, scores, NMS_IoU);              % NMS IoU (changeable)
    perFrameMs(k) = toc(t0)*1000;

    J = insertShape(I,'Rectangle',keepB,'LineWidth',2,'Color','red'); % det color (changeable)

    if haveGT
        key = strip_extension(frames(k).name);
        if isfield(GT,key)
            gtB = GT.(key);
            if ~isempty(gtB)
                J = insertShape(J,'Rectangle',gtB,'LineWidth',2,'Color','green'); % GT color (changeable)
                [tp,fp,fn] = match_detections(keepB, gtB, IoU_TP);                % IoU_TP (changeable)
                allTP=allTP+tp; allFP=allFP+fp; allFN=allFN+fn;
            end
        end
    end

    writeVideo(vw,J);
    % imshow(J); drawnow;                                          % live preview (optional)
end
close(vw);

% ---- METRICS + EXPORTS ----
fprintf('Saved video: %s | mean %.1f ms/frame (%.1f fps)\n', vidPath, mean(perFrameMs), 1000/mean(perFrameMs));
try, imwrite(J, fullfile(outFigureDir,'demo_detect_frame.png')); end          % figure export (changeable)

if haveGT
    prec = allTP / max(1, allTP+allFP);
    rec  = allTP / max(1, allTP+allFN);
    f1   = 2*prec*rec / max(1e-9,prec+rec);
    T = table(prec, rec, f1, mean(perFrameMs), 'VariableNames',{'Precision','Recall','F1','MsPerFrame'});
    writetable(T, fullfile(outTableDir,'detect_only_metrics.csv'));           % metrics path (changeable)
    fprintf('Metrics: P=%.3f R=%.3f F1=%.3f | %.1f ms/frame\n', prec, rec, f1, mean(perFrameMs));
else
    fprintf('GT not available → metrics CSV skipped.\n');
end
fprintf('DONE.\n');

% ---- helpers ----
function ensure_dir(varargin)
for i=1:nargin, d=varargin{i}; if ~exist(d,'dir'), mkdir(d); end, end
end
function s = strip_extension(fname)
[~,s,~] = fileparts(fname);
end
end
