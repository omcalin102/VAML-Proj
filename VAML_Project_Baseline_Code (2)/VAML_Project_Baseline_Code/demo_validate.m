function demo_validate()
% Produces CV table + a simple bar figure for slides (no detection run).

clc; close all; rng(42,'twister');                      % RNG seed (changeable)

% ---- PATHS ----
posDir   = fullfile('data','images','pos');             % pos path
negDir   = fullfile('data','images','neg');             % neg path
outTableDir  = fullfile('results','tables');            % tables out
outFigureDir = fullfile('report','figs');               % figs out
ensure_dir(outTableDir, outFigureDir);

% ---- HYPER-PARAMETERS ----
C_values  = [0.1 0.3 1 3 10];                           % SVM C grid (changeable)
splitMode = 'holdout';                                  % 'holdout' or 'kfold' (changeable)
kfoldK    = 5;                                          % K for k-fold (changeable)
primary   = 'F1';                                       % selection/display metric (changeable)

% ---- 1) DATASET ----
fprintf('[1/3] Building dataset ...\n');
[X,y] = build_dataset(posDir, negDir, 'FeatureType','hog');

% ---- 2) CV ----
fprintf('[2/3] Cross-validating ...\n');
if strcmpi(splitMode,'kfold')
    cvRes = crossval_eval(X, y, C_values, 'Split','kfold','K',kfoldK, ...
                          'OutDir',outTableDir,'PrimaryMetric',primary, 'ModelType','svm','ParamName','C');
else
    cvRes = crossval_eval(X, y, C_values, 'Split','holdout','Holdout',0.2, ...
                          'OutDir',outTableDir,'PrimaryMetric',primary, 'ModelType','svm','ParamName','C');  % holdout size (changeable)
end

% ---- 3) FIGURE (bar) ----
fprintf('[3/3] Plotting ...\n');
fig = figure('Color','w','Position',[100 100 900 420]);  % figure size (changeable)
metrics = {'Accuracy','Precision','Recall','F1'};
vals = table2array(cvRes(:, metrics));
bar(vals); grid on; box on;
xticklabels(compose('C=%.2g', cvRes.C));
legend(metrics, 'Location','southoutside','Orientation','horizontal');
title(sprintf('Cross-Validation (%s primary)', primary));
ylabel('Score'); ylim([0 1]);                           % ylim (changeable)

outFig = fullfile(outFigureDir, 'cv_baseline_bar.png');
exportgraphics(fig, outFig, 'Resolution', 150);         % resolution (changeable)
fprintf('Saved: %s\n', outFig);
close(fig);

fprintf('DONE.\n');

% ---- helpers ----
function ensure_dir(varargin)
for i=1:nargin, d = varargin{i}; if ~exist(d,'dir'), mkdir(d); end, end
end

end
