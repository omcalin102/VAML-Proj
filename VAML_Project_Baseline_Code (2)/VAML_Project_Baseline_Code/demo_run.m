function demo_run()

clc; close all; rng(42,'twister'); addpath(genpath('.')); % RNG seed (changeable)

% ---- PATHS ----
posDir   = fullfile('data','images','pos');         % training pos path
negDir   = fullfile('data','images','neg');         % training neg path
gtFile   = fullfile('data','test.dataset');         % GT file (optional)
frameDir = fullfile('pedestrian','pedestrian');     % test frames dir

outModelDir  = fullfile('results','models');        % models out
outTableDir  = fullfile('results','tables');        % tables out
outVideoDir  = fullfile('results','videos');        % video out
outFigureDir = fullfile('report','figs');           % figures out
ensure_dir(outModelDir, outTableDir, outVideoDir, outFigureDir);

% ---- HYPER-PARAMETERS ----
C_values    = [0.1 0.3 1 3 10];                     % SVM C grid (changeable)
splitMode   = 'holdout';                            % 'holdout' or 'kfold'
kfoldK      = 5;                                    % K for k-fold (changeable)
BaseWindow  = [64 128];                             % detector window (changeable)
Step        = 8;                                    % stride px (4/8/12) (changeable)
ScaleFactor = 0.90;                                 % pyramid factor (0.85–0.95) (changeable)
NMS_IoU     = 0.30;                                 % NMS IoU (0.3–0.5) (changeable)
MinScore    = -Inf;                                 % pre-NMS score filter (changeable)
MaxFrames   = 10;                                   % frames to process (changeable)
IoU_TP      = 0.50;                                 % TP IoU rule (changeable)

% ---- 1) DATASET ----
fprintf('\n[1/5] Building dataset ...\n');
[X,y] = build_dataset(posDir, negDir);              % feature builder

% ---- 2) CV ----
fprintf('[2/5] Cross-validating SVM ...\n');
if strcmpi(splitMode,'kfold')                        % choose k-fold vs holdout
    cvRes = crossval_eval(X, y, C_values, 'Split','kfold','K',kfoldK, ...
                          'OutDir',outTableDir,'PrimaryMetric','F1');
else
    cvRes = crossval_eval(X, y, C_values, 'Split','holdout','Holdout',0.2, ...
                          'OutDir',outTableDir,'PrimaryMetric','F1');  % 0.2 holdout (changeable)
end
[~,ixBest] = max(cvRes(:,5));                        % select by F1 (changeable metric)
bestC = cvRes(ixBest,1);
fprintf('Best C=%.3f (F1=%.3f, Acc=%.3f)\n', bestC, cvRes(ixBest,5), cvRes(ixBest,2));

% ---- 3) TRAIN ----
fprintf('[3/5] Training best model ...\n');
model = train_svm(X, y, bestC);                      % training call
save(fullfile(outModelDir,'model_baseline.mat'), 'model');  % model path

% ---- 4) DETECT + NMS + VIDEO ----
fprintf('[4/5] Running detector ...\n');
frames = dir(fullfile(frameDir, '*.jpg'));
if isempty(frames), frames = dir(fullfile(frameDir, '*.png')); end
frames = frames(1:min(MaxFrames, numel(frames)));    % limit frames (changeable)

vidPath = fullfile(outVideoDir, 'demo_baseline.mp4'); % video name (changeable)
try
    vw = VideoWriter(vidPath, 'MPEG-4');                % MP4 (best if supported)
catch
    vidPath = fullfile(outVideoDir, 'demo_baseline.avi'); % fallback filename
    vw = VideoWriter(vidPath, 'Motion JPEG AVI');         % AVI fallback
end
vw.FrameRate = 6;  % changeable
open(vw);

haveGT = false; GT = [];
if exist(gtFile,'file') == 2 && exist('load_gt','file') == 2
    try
        GT = load_gt(gtFile);                        % ensure returns struct/map name->Nx4
        haveGT = true;
    catch, haveGT = false;
    end
end

allTP=0; allFP=0; allFN=0;
loopTimer = tic;
for k = 1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));

    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',BaseWindow, ...                 % window (changeable)
        'Step',Step, ...                             % stride (changeable)
        'ScaleFactor',ScaleFactor, ...               % pyramid factor (changeable)
        'MinScore',MinScore);                        % pre-filter (changeable)

    [keepB, keepS] = nms(boxes, scores, NMS_IoU);    % NMS IoU (changeable)

    J = insertShape(I, 'Rectangle', keepB, 'LineWidth', 2, 'Color', 'red');  % draw color (changeable)

    if haveGT
        key = strip_extension(frames(k).name);
        if isfield(GT, key)
            gtB = GT.(key);
            if ~isempty(gtB)
                J = insertShape(J, 'Rectangle', gtB, 'LineWidth', 2, 'Color', 'green'); % GT color (changeable)
                [tp,fp,fn] = match_detections(keepB, gtB, IoU_TP);  % IoU_TP (changeable)
                allTP = allTP + tp; allFP = allFP + fp; allFN = allFN + fn;
            end
        end
    end

    writeVideo(vw, J);
    % imshow(J); drawnow;                          % enable live preview (optional)
    progress_bar(k, numel(frames), loopTimer, '  - detecting');
end
t = toc(loopTimer); close(vw);
fprintf('Saved: %s | %.2fs total (~%.2f fps)\n', vidPath, t, max(eps, numel(frames)/t));

% ---- 5) METRICS ----
if haveGT
    prec = allTP / max(1, allTP + allFP);
    rec  = allTP / max(1, allTP + allFN);
    f1   = 2*prec*rec / max(1e-9, (prec+rec));
    DT = table(bestC, prec, rec, f1, 'VariableNames',{'C','Precision','Recall','F1'});
    writetable(DT, fullfile(outTableDir,'detection_baseline.csv'));  % output path (changeable)
else
    fprintf('No GT per-frame → skipped detection metrics.\n');
end

% ---- EXPORT FIGURE ----
try
    imwrite(J, fullfile(outFigureDir,'demo_detection_frame.png'));   % figure path (changeable)
catch
end

fprintf('DONE.\n');

% ---- helpers ----
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

end
