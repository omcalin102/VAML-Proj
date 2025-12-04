function eval_video(varargin)
% Annotate a video with pedestrian detections; save MP4.
%
% Example:
% eval_video('VideoPath','data/shopping_center.mpg', ...
%   'ModelPath','results/models/model_baseline.mat', 'RunTag','run3');

p = inputParser;
addParameter(p,'VideoPath','');                                  % <- set this
addParameter(p,'ModelPath',fullfile('results','models','model_baseline.mat'));
addParameter(p,'BaseWindow',[64 128]);                           % [w h]
addParameter(p,'Step',16);                                       % stride
addParameter(p,'ResizeLongSide',480);                            % work size
addParameter(p,'MinScore',0.6);                                  % threshold
addParameter(p,'NMS_IoU',0.25);                                  % NMS
addParameter(p,'RunTag','run3');                                 % subfolder
parse(p,varargin{:}); a = p.Results;

assert(exist(a.VideoPath,'file')==2,'Video not found: %s', a.VideoPath);
assert(exist(a.ModelPath,'file')==2,'Model not found: %s', a.ModelPath);

outDir = fullfile('results','videos',a.RunTag);
if ~exist(outDir,'dir'), mkdir(outDir); end

[~,stem,~] = fileparts(a.VideoPath);
outMp4 = fullfile(outDir, sprintf('%s_%s.mp4', stem, a.RunTag));

S = load(a.ModelPath); model = S.model;
vr = VideoReader(a.VideoPath);
vw = VideoWriter(outMp4,'MPEG-4'); vw.FrameRate = vr.FrameRate; open(vw);

fprintf('Annotating %s → %s\n', a.VideoPath, outMp4);
f = 0;
while hasFrame(vr)
    f = f+1;
    I0 = readFrame(vr);
    [H,W,~] = size(I0); longSide = max(H,W);

    scale=1.0; I=I0; stepWork=a.Step;
    if a.ResizeLongSide>0 && longSide>a.ResizeLongSide
        scale = a.ResizeLongSide/double(longSide);
        I = imresize(I0, scale);
        stepWork = max(4, round(a.Step*scale));
    end

    [B,S] = score_windows(I,model,'BaseWindow',a.BaseWindow,'Step',stepWork, ...
                          'MinScore',a.MinScore,'MaxWindows',200,'Verbose',false);
    [B,S] = nms_fast(B,S,a.NMS_IoU);
    % geometric post-filter (same as images)
    [B,S] = box_postfilter(B,S,'Aspect',[0.35 0.60], ...
        'MinH', round(0.28*size(I,1)), 'TopFrac', 0.30);
    if scale~=1.0, B = round(B./scale); end

    % draw
    J = insertShape(I0,'Rectangle',B,'Color','red','LineWidth',2);
    if ~isempty(B)
        lbl = arrayfun(@(s) sprintf('%.2f',s), S,'UniformOutput',false);
        anc = [B(:,1)+1, max(1, B(:,2)-12)];
        J = insertText(J, anc, lbl, 'FontSize', 12, 'BoxOpacity',0.5, ...
                       'BoxColor','red','TextColor','white');
    end

    % basic progress
    if mod(f,10)==0, fprintf('  frame %d\n', f); end
    writeVideo(vw,J);
end
close(vw);
fprintf('Saved video: %s\n', outMp4);
end

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
