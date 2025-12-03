function demo_run()

clc; close all; rng(42,'twister');                  % RNG seed (changeable)

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

% ---- FEATURE / MODEL OPTIONS ----
ResizeTo     = [64 128];                             % detector window (changeable)
CellSize     = [8 8];                                % HOG cell (changeable)
BlockSize    = [2 2];                                % HOG block in cells (changeable)
BlockOverlap = [1 1];                                % HOG block overlap (changeable)
NumBins      = 9;                                    % HOG bins (changeable)

featureConfigs = [ ...
    struct('Name','HOG',        'Args', {{'FeatureType','hog','ResizeTo',ResizeTo,'CellSize',CellSize,'BlockSize',BlockSize,'BlockOverlap',BlockOverlap,'NumBins',NumBins}}), ...
    struct('Name','HOG+PCA128', 'Args', {{'FeatureType','hog','ResizeTo',ResizeTo,'CellSize',CellSize,'BlockSize',BlockSize,'BlockOverlap',BlockOverlap,'NumBins',NumBins,'PCAComponents',128}}), ...
    struct('Name','Pixels+PCA', 'Args', {{'FeatureType','pixels','ResizeTo',ResizeTo,'PCAComponents',0.90}}) ...
    ];

modelConfigs = [ ...
    struct('ModelType','svm', 'ParamName','C',            'Grid',[0.1 0.3 1 3 10]), ...
    struct('ModelType','knn', 'ParamName','NumNeighbors', 'Grid',[3 5 7]) ...
    ];

splitMode   = 'holdout';                            % 'holdout' or 'kfold'
kfoldK      = 5;                                    % K for k-fold (changeable)
Step        = 8;                                    % stride px (4/8/12) (changeable)
ScaleFactor = 0.90;                                 % pyramid factor (0.85–0.95) (changeable)
NMS_IoU     = 0.30;                                 % NMS IoU (0.3–0.5) (changeable)
MinScore    = -Inf;                                 % pre-NMS score filter (changeable)
MaxFrames   = 10;                                   % frames to process (changeable)
IoU_TP      = 0.50;                                 % TP IoU rule (changeable)

% ---- 1) DATASET + GRID SEARCH ----
fprintf('\n[1/5] Building datasets + CV grid ...\n');
summaryRows = {};
best = struct('F1', -inf);

for f = 1:numel(featureConfigs)
    fCfg = featureConfigs(f);
    [X,y,descCfg] = build_dataset(posDir, negDir, fCfg.Args{:});

    for m = 1:numel(modelConfigs)
        mCfg = modelConfigs(m);
        if strcmpi(splitMode,'kfold')
            cvRes = crossval_eval(X,y,mCfg.Grid,'Split','kfold','K',kfoldK, ...
                'OutDir',outTableDir,'Label',sprintf('%s_%s',lower(fCfg.Name),mCfg.ModelType), ...
                'PrimaryMetric','F1','ModelType',mCfg.ModelType,'ParamName',mCfg.ParamName);
        else
            cvRes = crossval_eval(X,y,mCfg.Grid,'Split','holdout','Holdout',0.2, ...
                'OutDir',outTableDir,'Label',sprintf('%s_%s',lower(fCfg.Name),mCfg.ModelType), ...
                'PrimaryMetric','F1','ModelType',mCfg.ModelType,'ParamName',mCfg.ParamName);
        end

        top = cvRes(1,:);
        summaryRows(end+1,:) = {fCfg.Name, mCfg.ModelType, top.(mCfg.ParamName), top.Accuracy, top.Precision, top.Recall, top.F1}; %#ok<AGROW>
        if top.F1 > best.F1
            best = struct('FeatureName', fCfg.Name, ...
                'Descriptor', descCfg, ...
                'ModelType', mCfg.ModelType, ...
                'ParamName', mCfg.ParamName, ...
                'ParamValue', top.(mCfg.ParamName), ...
                'F1', top.F1, 'Accuracy', top.Accuracy, 'Precision', top.Precision, 'Recall', top.Recall, ...
                'X', X, 'y', y);
        end
    end
end

summaryTable = cell2table(summaryRows, 'VariableNames', {'Feature','Model','ParamValue','Accuracy','Precision','Recall','F1'});
writetable(summaryTable, fullfile(outTableDir, 'cv_summary_grid.csv'));

% ---- 2) TRAIN BEST MODEL ----
fprintf('[2/5] Training best model: %s + %s=%.3g (F1=%.3f) ...\n', best.FeatureName, best.ParamName, best.ParamValue, best.F1);
classifier = train_model(best.X, best.y, best.ModelType, best.ParamValue, 'Standardize', true, 'ClassNames', unique(best.y));
model = struct('Classifier', classifier, 'Descriptor', best.Descriptor, ...
    'ModelType', best.ModelType, 'ParamName', best.ParamName, 'ParamValue', best.ParamValue, ...
    'FeatureName', best.FeatureName, 'CV', rmfield(best, {'X','y','Descriptor'}));
save(fullfile(outModelDir,'model_best_combo.mat'), 'model');

% ---- 3) DETECT + NMS + VIDEO ----
fprintf('[3/5] Running detector ...\n');
frames = dir(fullfile(frameDir, '*.jpg'));
if isempty(frames), frames = dir(fullfile(frameDir, '*.png')); end
frames = frames(1:min(MaxFrames, numel(frames)));    % limit frames (changeable)

vidPath = fullfile(outVideoDir, 'demo_best_combo.mp4');
try
    vw = VideoWriter(vidPath, 'MPEG-4');
catch
    vidPath = fullfile(outVideoDir, 'demo_best_combo.avi');
    vw = VideoWriter(vidPath, 'Motion JPEG AVI');
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
tic;
for k = 1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));

    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',best.Descriptor.ResizeTo, ...   % window (changeable)
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
end
t = toc; close(vw);
fprintf('Saved: %s | %.2fs total (~%.2f fps)\n', vidPath, t, max(eps, numel(frames)/t));

% ---- 4) METRICS ----
if haveGT
    prec = allTP / max(1, allTP + allFP);
    rec  = allTP / max(1, allTP + allFN);
    f1   = 2*prec*rec / max(1e-9, (prec+rec));
    DT = table(best.ParamValue, prec, rec, f1, 'VariableNames',{'ParamValue','Precision','Recall','F1'});
    writetable(DT, fullfile(outTableDir,'detection_best_combo.csv'));  % output path (changeable)
else
    fprintf('No GT per-frame → skipped detection metrics.\n');
end

% ---- 5) EXPORT FIGURE ----
try
    imwrite(J, fullfile(outFigureDir,'demo_detection_frame.png'));   % figure path (changeable)
catch
end

fprintf('DONE.\n');

% ---- helpers ----
function ensure_dir(varargin)
for i=1:nargin, d = varargin{i}; if ~exist(d,'dir'), mkdir(d); end, end
end
function s = strip_extension(fname)
[~,s,~] = fileparts(fname);
end

end
