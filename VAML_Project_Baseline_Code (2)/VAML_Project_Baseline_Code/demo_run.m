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
MinScore    = 0;                                    % pre-NMS score filter (changeable)
FPS         = 6;                                    % output video fps (changeable)
VideoQuality= 100;                                  % video quality 0–100 (changeable)
MaxFrames   = 10;                                   % frames to process (changeable)
IoU_TP      = 0.50;                                 % TP IoU rule (changeable)

% ---- 1) DATASET + GRID SEARCH ----
fprintf('\n[1/6] Building datasets + CV grid ...\n');
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
fprintf('[2/6] Training best model: %s + %s=%.3g (F1=%.3f) ...\n', best.FeatureName, best.ParamName, best.ParamValue, best.F1);
classifier = train_model(best.X, best.y, best.ModelType, best.ParamValue, 'Standardize', true, 'ClassNames', unique(best.y));
model = struct('Classifier', classifier, 'Descriptor', best.Descriptor, ...
    'ModelType', best.ModelType, 'ParamName', best.ParamName, 'ParamValue', best.ParamValue, ...
    'FeatureName', best.FeatureName, 'CV', rmfield(best, {'X','y','Descriptor'}));
save(fullfile(outModelDir,'model_best_combo.mat'), 'model');

% ---- 3) CLASSIFIER METRICS + VISUALS ----
fprintf('[3/6] Cross-validating best model for metrics/plots ...\n');
try
    cvmdl = crossval(classifier, 'KFold', kfoldK);
    [cvPreds, cvScores] = kfoldPredict(cvmdl);

    posScores = extract_positive_scores(cvScores, classifier);
    [accCv, precCv, recCv, f1Cv, cmCv] = pr_metrics(cvPreds, best.y);

    classMetrics = table(best.ParamValue, accCv, precCv, recCv, f1Cv, ...
        'VariableNames',{'ParamValue','Accuracy','Precision','Recall','F1'});
    writetable(classMetrics, fullfile(outTableDir, 'classifier_best_combo.csv'));

    classConfPath = fullfile(outFigureDir, 'classifier_confusion_heatmap.png');
    save_confusion_heatmap(cmCv, {'Positive','Negative'}, classConfPath, ...
        sprintf('Confusion Matrix (%s)', best.FeatureName));

    if ~isempty(posScores)
        % ROC curve
        [fpr, tpr, ~, aucRoc] = perfcurve(best.y, posScores, 1);
        rocPath = fullfile(outFigureDir, 'classifier_roc_curve.png');
        save_curve_plot(fpr, tpr, 'False Positive Rate', 'True Positive Rate', ...
            sprintf('ROC (%s, AUC=%.3f)', best.FeatureName, aucRoc), rocPath);

        % Precision-Recall curve
        [recVals, precVals, ~, aucPr] = perfcurve(best.y, posScores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
        prPath = fullfile(outFigureDir, 'classifier_pr_curve.png');
        save_curve_plot(recVals, precVals, 'Recall', 'Precision', ...
            sprintf('Precision-Recall (%s, AUC=%.3f)', best.FeatureName, aucPr), prPath);
    end
catch ME
    warning('Skipping classifier metric visuals: %s', ME.message);
end

% ---- 4) DETECT + NMS + VIDEO ----
fprintf('[4/6] Running detector ...\n');
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
vw.FrameRate = FPS;  % changeable
if isprop(vw,'Quality')
    vw.Quality = VideoQuality;                      % clamp compression to max quality (changeable)
end
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
perFrameMs = zeros(numel(frames),1);
perFrameTP = zeros(numel(frames),1);
perFrameFP = zeros(numel(frames),1);
perFrameFN = zeros(numel(frames),1);

lastFrame = [];
tic;
for k = 1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));

    t0 = tic;
    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',best.Descriptor.ResizeTo, ...   % window (changeable)
        'Step',Step, ...                             % stride (changeable)
        'ScaleFactor',ScaleFactor, ...               % pyramid factor (changeable)
        'MinScore',MinScore);                        % pre-filter (changeable)

    [keepB, keepS] = nms(boxes, scores, NMS_IoU);    % NMS IoU (changeable)
    perFrameMs(k) = toc(t0) * 1000;

    posMask = keepS > MinScore;                      % keep only positive detections
    keepB = keepB(posMask,:);

    J = insertShape(I, 'Rectangle', keepB, 'LineWidth', 2, 'Color', 'red');  % draw color (changeable)

    if haveGT
        key = strip_extension(frames(k).name);
        if isfield(GT, key)
            gtB = GT.(key);
            if ~isempty(gtB)
                J = insertShape(J, 'Rectangle', gtB, 'LineWidth', 2, 'Color', 'green'); % GT color (changeable)
                [tp,fp,fn] = match_detections(keepB, gtB, IoU_TP);  % IoU_TP (changeable)
                allTP = allTP + tp; allFP = allFP + fp; allFN = allFN + fn;
                perFrameTP(k) = tp; perFrameFP(k) = fp; perFrameFN(k) = fn;
            end
        end
    end

    writeVideo(vw, J);
    lastFrame = J;
    % imshow(J); drawnow;                          % enable live preview (optional)
end
t = toc; close(vw);
fprintf('Saved: %s | %.2fs total (~%.2f fps)\n', vidPath, t, max(eps, numel(frames)/t));
fprintf('Mean frame time: %.1f ms/frame | output fps=%g quality=%g\n', mean(perFrameMs), vw.FrameRate, getfield_safe(vw,'Quality'));

% ---- 5) METRICS + VISUALS ----
fprintf('[5/6] Computing detection metrics ...\n');
if haveGT
    prec = allTP / max(1, allTP + allFP);
    rec  = allTP / max(1, allTP + allFN);
    f1   = 2*prec*rec / max(1e-9, (prec+rec));
    DT = table(best.ParamValue, prec, rec, f1, 'VariableNames',{'ParamValue','Precision','Recall','F1'});
    writetable(DT, fullfile(outTableDir,'detection_best_combo.csv'));  % output path (changeable)

    % Per-frame metrics visualization
    frameIds = (1:numel(frames))';
    pfPrec = perFrameTP ./ max(1, perFrameTP + perFrameFP);
    pfRec  = perFrameTP ./ max(1, perFrameTP + perFrameFN);

    detMetricsPath = fullfile(outFigureDir, 'detection_per_frame.png');
    save_detection_metrics_plot(frameIds, perFrameTP, perFrameFP, perFrameFN, pfPrec, pfRec, detMetricsPath);

    detConfPath = fullfile(outFigureDir, 'detection_confusion_heatmap.png');
    detCM = [allTP allFP; allFN 0];
    save_confusion_heatmap(detCM, {'Detection','Background'}, detConfPath, 'Detection Confusion');
else
    fprintf('No GT per-frame → skipped detection metrics.\n');
end

% ---- 6) EXPORT FIGURE ----
fprintf('[6/6] Exporting demo frame ...\n');
try
    if ~isempty(lastFrame)
        imwrite(lastFrame, fullfile(outFigureDir,'demo_detection_frame.png'));   % figure path (changeable)
    end
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
function v = getfield_safe(S, field)
if isprop(S, field)
    v = S.(field);
else
    v = NaN;
end
end
function scores = extract_positive_scores(scores, classifier)
if isempty(scores)
    return;
end
if isstruct(classifier) && isfield(classifier,'Classifier')
    cls = classifier.Classifier;
else
    cls = classifier;
end
try
    names = string(cls.ClassNames);
    posIx = find(names=="1" | lower(names)=="pos" | lower(names)=="positive" | names=="true", 1);
    if isempty(posIx), posIx = size(scores,2); end
    scores = scores(:, posIx);
catch
    scores = scores(:);
end
end
function save_confusion_heatmap(cm, labels, outPath, ttl)
try
    figure('Visible','off');
    h = heatmap(labels, labels, cm);
    h.Title = ttl;
    h.CellLabelFormat = '%.0f';
    xlabel('Predicted'); ylabel('True');
    saveas(gcf, outPath);
    close(gcf);
catch ME
    warning('Could not save confusion heatmap: %s', ME.message);
end
end
function save_curve_plot(x, y, xlab, ylab, ttl, outPath)
try
    figure('Visible','off');
    plot(x, y, 'LineWidth', 2);
    grid on; xlim([0 1]); ylim([0 1]);
    xlabel(xlab); ylabel(ylab); title(ttl);
    saveas(gcf, outPath);
    close(gcf);
catch ME
    warning('Could not save curve plot: %s', ME.message);
end
end
function save_detection_metrics_plot(frameIds, tp, fp, fn, prec, rec, outPath)
try
    figure('Visible','off');
    tiledlayout(2,1);

    nexttile;
    bar(frameIds, [tp fp fn], 'stacked');
    legend({'TP','FP','FN'},'Location','northoutside','Orientation','horizontal');
    ylabel('Count'); xlabel('Frame #');
    title('Per-frame Detection Counts');

    nexttile;
    plot(frameIds, prec, '-o', 'LineWidth', 1.5); hold on;
    plot(frameIds, rec, '-s', 'LineWidth', 1.5);
    ylim([0 1]); grid on;
    xlabel('Frame #'); ylabel('Score');
    legend({'Precision','Recall'},'Location','southoutside','Orientation','horizontal');
    title('Per-frame Precision/Recall');

    saveas(gcf, outPath);
    close(gcf);
catch ME
    warning('Could not save detection metrics plot: %s', ME.message);
end
end

end
