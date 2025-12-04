function eval_on_imageset(varargin)
% Run detector on up to N images; save annotated PNGs + CSV.
% Saves to: <OutDir>/<RunTag>/ to avoid overwriting previous runs.

p = inputParser;
addParameter(p,'ModelPath',fullfile('results','models','model_baseline.mat'));  % trained SVM path
addParameter(p,'ImageDir','');                                                  % folder with JPG/PNG
addParameter(p,'MaxImages',5);                                                  % how many images to run
addParameter(p,'BaseWindow',[64 128]);                                          % detector window [w h]
addParameter(p,'Step',16);                                                      % stride pixels
addParameter(p,'ResizeLongSide',480);                                           % downsize long side (speed)
addParameter(p,'MinScore',0.6);                                                 % margin threshold (stricter)
addParameter(p,'NMS_IoU',0.25);                                                 % NMS IoU (lower = fewer merges)
addParameter(p,'OutDir',fullfile('results','frames_sample'));                   % base output dir
addParameter(p,'RunTag','run2');                                                % subfolder (avoid overwrite)
addParameter(p,'KeepTopK',5);                                                   % keep top-K detections per image
parse(p,varargin{:}); a = p.Results;

assert(~isempty(a.ImageDir) && exist(a.ImageDir,'dir')==7, 'Bad ImageDir: %s', a.ImageDir);
assert(exist(a.ModelPath,'file')==2, 'Model not found: %s', a.ModelPath);

% final output folder
outDirFinal = fullfile(a.OutDir, a.RunTag);
if ~exist(outDirFinal,'dir'), mkdir(outDirFinal); end

S = load(a.ModelPath); model = S.model;

% image list
imgs = [dir(fullfile(a.ImageDir,'*.jpg')); dir(fullfile(a.ImageDir,'*.png')); ...
        dir(fullfile(a.ImageDir,'*.JPG')); dir(fullfile(a.ImageDir,'*.PNG'))];
assert(~isempty(imgs),'No JPG/PNG found in %s', a.ImageDir);
N = min(a.MaxImages, numel(imgs)); imgs = imgs(1:N);

fprintf('Processing %d image(s) from %s → %s\n', N, a.ImageDir, outDirFinal);
rows = cell(0,3);

for k = 1:N
    name  = imgs(k).name;
    inpth = fullfile(imgs(k).folder, name);
    fprintf('[%d/%d] %s\n', k, N, name); drawnow;

    I0 = imread(inpth);
    [H,W,~] = size(I0); longSide = max(H,W);

    % resize for speed
    scale = 1.0; I = I0; stepWork = a.Step;
    if a.ResizeLongSide>0 && longSide>a.ResizeLongSide
        scale = a.ResizeLongSide / double(longSide);
        I = imresize(I0, scale);
        stepWork = max(4, round(a.Step * scale));      % stride proportional to scale
    end

    t0 = tic;
    [Bw,Sc] = score_windows(I, model, ...
        'BaseWindow', a.BaseWindow, ...   % ← change window size here
        'Step', stepWork, ...             % ← change stride here
        'MinScore', a.MinScore, ...       % ← change threshold here
        'MaxWindows', 200, ...            % cap windows for speed
        'Verbose', true);
    [Bw,Sc] = nms_fast(Bw, Sc, a.NMS_IoU);            % single NMS

    % geometric post-filter to cut poles/trees & tiny boxes
    [Bw,Sc] = box_postfilter(Bw, Sc, ...
        'Aspect',[0.35 0.60], ...                     % ← w/h range for upright people
        'MinH', round(0.28*size(I,1)), ...            % ← min height (relative to image)
        'TopFrac', 0.30);                              % ← ignore top 30% (signs/awnings)

    % keep top-K detections for clean visuals (optional)
    if a.KeepTopK > 0 && numel(Sc) > a.KeepTopK
        [~, ord] = maxk(Sc, a.KeepTopK);
        Bw = Bw(ord,:);  Sc = Sc(ord);
    end
    ms = toc(t0)*1000;

    % map boxes back to original size
    if scale~=1.0, B = round(Bw ./ scale); else, B = Bw; end

    % draw and save
    J = insertShape(I0,'Rectangle',B,'Color','red','LineWidth',2);
    if ~isempty(B)
        lbl = arrayfun(@(s) sprintf('%.2f',s), Sc,'UniformOutput',false);
        anc = [B(:,1)+1, max(1, B(:,2)-12)];
        J = insertText(J, anc, lbl, 'FontSize', 12, 'BoxOpacity',0.5, ...
                       'BoxColor','red','TextColor','white');
    end
    stem = erase(name,{'.jpg','.png','.JPG','.PNG'});
    outpng = fullfile(outDirFinal, sprintf('%s_%s_det.png', stem, a.RunTag));
    imwrite(J, outpng);
    fprintf('  saved %s (%.1f ms)\n', outpng, ms);

    rows(end+1,:) = {name, size(B,1), ms}; %#ok<AGROW>
end

% small CSV summary (filename, #dets, ms)
T = cell2table(rows,'VariableNames',{'Image','NumDetections','MsPerImage'});
writetable(T, fullfile(outDirFinal, sprintf('image_detection_%s.csv', a.RunTag)));
fprintf('Saved PNGs + CSV to %s\n', outDirFinal);
end

% -------- simple geometric post-filter --------
function [B,S] = box_postfilter(B,S,varargin)
p = inputParser;
addParameter(p,'Aspect',[0.3 0.8]);     % valid aspect (w/h)
addParameter(p,'MinH',60);              % min height (px)
addParameter(p,'TopFrac',0.2);          % ignore boxes whose top is in top X%
parse(p,varargin{:}); a=p.Results;
if isempty(B), return; end
w=B(:,3); h=B(:,4); ar=w./max(1,h);
imgH = max(B(:,2)+B(:,4));              % rough image height from boxes
keep = ar>=a.Aspect(1) & ar<=a.Aspect(2) & ...
       h>=a.MinH & ...
       (B(:,2) > a.TopFrac*imgH);
B=B(keep,:); S=S(keep);
end
