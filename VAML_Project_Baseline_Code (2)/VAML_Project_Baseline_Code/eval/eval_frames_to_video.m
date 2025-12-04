function eval_frames_to_video(varargin)
% Turn an image sequence into an annotated video using the trained detector.
% Auto-selects a supported VideoWriter profile (MPEG-4 → fallback to Motion JPEG AVI).

p = inputParser;
addParameter(p,'ImageDir','');
addParameter(p,'ModelPath',fullfile('results','models','model_baseline.mat'));
addParameter(p,'RunTag','run4');
addParameter(p,'FPS',10);
addParameter(p,'BaseWindow',[64 128]);
addParameter(p,'Step',16);
addParameter(p,'ResizeLongSide',480);
addParameter(p,'MinScore',0.6);
addParameter(p,'NMS_IoU',0.25);
parse(p,varargin{:}); a = p.Results;

assert(~isempty(a.ImageDir) && exist(a.ImageDir,'dir')==7, 'Bad ImageDir: %s', a.ImageDir);
assert(exist(a.ModelPath,'file')==2, 'Model not found: %s', a.ModelPath);

% Output folder (absolute, derived from model path)
projRoot = fileparts(fileparts(a.ModelPath));  % .../VAML_Project_Baseline_Code
outDir = fullfile(projRoot,'results','videos',a.RunTag);
if ~exist(outDir,'dir'), mkdir(outDir); end

% Collect frames and sort by name
imgs = [dir(fullfile(a.ImageDir,'*.jpg')); dir(fullfile(a.ImageDir,'*.png')); ...
        dir(fullfile(a.ImageDir,'*.JPG')); dir(fullfile(a.ImageDir,'*.PNG'))];
assert(~isempty(imgs),'No JPG/PNG found in %s', a.ImageDir);
[~,ord] = sort({imgs.name}); imgs = imgs(ord);

[~,stem] = fileparts(a.ImageDir);

% --- pick a supported video profile and extension ---
[profile, ext] = pick_writer_profile();                            % <-- NEW
outVid = fullfile(outDir, sprintf('%s_%s%s', stem, a.RunTag, ext));

% Load model
S = load(a.ModelPath); model = S.model;

% Video size from first frame
I0 = imread(fullfile(imgs(1).folder, imgs(1).name));
[H0,W0,~] = size(I0);

% Create writer
vw = VideoWriter(outVid, profile);
vw.FrameRate = a.FPS;
open(vw);

fprintf('Building video (%s) from %d frame(s) → %s\n', profile, numel(imgs), outVid);

for k=1:numel(imgs)
    I0 = imread(fullfile(imgs(k).folder, imgs(k).name));
    [H,W,~] = size(I0); longSide = max(H,W);

    % Work image resize for speed
    scale=1.0; I=I0; stepWork=a.Step;
    if a.ResizeLongSide>0 && longSide>a.ResizeLongSide
        scale = a.ResizeLongSide/double(longSide);
        I = imresize(I0, scale);
        stepWork = max(4, round(a.Step*scale));
    end

    % Detect
    [B,S] = score_windows(I, model, 'BaseWindow',a.BaseWindow, ...
                          'Step',stepWork, 'MinScore',a.MinScore, ...
                          'MaxWindows',200,'Verbose',false);
    [B,S] = nms_fast(B,S,a.NMS_IoU);
    [B,S] = box_postfilter(B,S,'Aspect',[0.35 0.60], ...
        'MinH', round(0.28*size(I,1)), 'TopFrac',0.30);
    if scale~=1.0, B = round(B./scale); end

    % Draw
    J = insertShape(I0,'Rectangle',B,'Color','red','LineWidth',2);
    if ~isempty(B)
        lbl = arrayfun(@(s) sprintf('%.2f',s), S,'UniformOutput',false);
        anc = [B(:,1)+1, max(1, B(:,2)-12)];
        J = insertText(J, anc, lbl, 'FontSize', 12, 'BoxOpacity',0.5, ...
                       'BoxColor','red','TextColor','white');
    end

    % Ensure stable video size
    if size(J,1)~=H0 || size(J,2)~=W0
        J = imresize(J,[H0 W0]);
    end

    if mod(k,10)==0, fprintf('  frame %d/%d\n', k, numel(imgs)); end
    writeVideo(vw,J);
end
close(vw);
fprintf('Saved: %s\n', outVid);
end

% --- choose a writer profile that works in this MATLAB environment ---
function [profile, ext] = pick_writer_profile()
% Prefer MPEG-4; if unsupported (MATLAB Online/Linux), fall back to Motion JPEG AVI.
try
    vwTest = VideoWriter(tempname, 'MPEG-4'); %#ok<NASGU>
    profile = 'MPEG-4'; ext = '.mp4';
catch
    profile = 'Motion JPEG AVI'; ext = '.avi';
end
end

% --- tiny post-filter to reduce poles/trees & tiny boxes ---
function [B,S] = box_postfilter(B,S,varargin)
p = inputParser;
addParameter(p,'Aspect',[0.3 0.8]);
addParameter(p,'MinH',60);
addParameter(p,'TopFrac',0.2);
parse(p,varargin{:}); a=p.Results;
if isempty(B), return; end
w=B(:,3); h=B(:,4); ar=w./max(1,h);
imgH = max(B(:,2)+B(:,4));
keep = ar>=a.Aspect(1) & ar<=a.Aspect(2) & h>=a.MinH & (B(:,2) > a.TopFrac*imgH);
B=B(keep,:); S=S(keep);
end
