function demo_track_by_detection()
%DEMO_TRACK_BY_DETECTION Simple tracker built on the sliding-window detector.
%   Runs detection on each frame, links boxes across time using IoU matching
%   (tracking-by-detection) and writes an annotated video with track IDs.

clc; close all; rng(42,'twister');                           % RNG seed (changeable)

% ---- PATHS ----
modelPath   = fullfile('results','models','model_best_combo.mat'); % model path (changeable)
frameDir    = fullfile('pedestrian','pedestrian');                % test frames dir (changeable)
outVideoDir = fullfile('results','videos');                       % video out (changeable)
outFigureDir= fullfile('report','figs');                          % figs out (changeable)
ensure_dir(outVideoDir, outFigureDir);

% ---- DETECTOR HYPER-PARAMETERS ----
BaseWindow   = [64 128];        % window size [w h] (changeable, overridden by model descriptor)
Step         = 8;               % stride px (changeable)
ScaleFactor  = 0.90;            % pyramid factor (changeable)
NMS_IoU      = 0.30;            % NMS IoU (changeable)
MinScore     = 0;               % pre-NMS score filter (changeable)
MaxFrames    = 30;              % frames to process (changeable)
FPS          = 6;               % output video fps (changeable)
VideoQuality = 100;             % video quality 0–100 (changeable)

% ---- TRACKER HYPER-PARAMETERS ----
AssignIoU = 0.30;               % IoU threshold for track association (changeable)
MaxMissing = 5;                 % drop track after this many missed frames (changeable)

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
frames = frames(1:min(MaxFrames, numel(frames)));

% ---- VIDEO WRITER ----
vidPath = fullfile(outVideoDir,'demo_track_by_detection.mp4');
try
    vw = VideoWriter(vidPath, 'MPEG-4');
catch
    vidPath = fullfile(outVideoDir, 'demo_track_by_detection.avi');
    vw = VideoWriter(vidPath, 'Motion JPEG AVI');
end
vw.FrameRate = FPS;
if isprop(vw,'Quality')
    vw.Quality = VideoQuality;
end
open(vw);

% ---- RUN DETECTOR + TRACKER ----
tracks = struct('id', {}, 'bbox', {}, 'age', {}, 'miss', {}, 'frames', {}, 'score', {});
nextId = 1;
perFrameMs = zeros(numel(frames),1);

for k=1:numel(frames)
    I = imread(fullfile(frames(k).folder, frames(k).name));
    t0 = tic;
    [boxes, scores] = score_windows(I, model, ...
        'BaseWindow',BaseWindow, ...
        'Step',Step, ...
        'ScaleFactor',ScaleFactor, ...
        'MinScore',MinScore, ...
        'Verbose', false);
    [keepB, keepS] = nms(boxes, scores, NMS_IoU);
    perFrameMs(k) = toc(t0)*1000;

    posMask = keepS > MinScore;
    detB = keepB(posMask,:);
    detS = keepS(posMask);

    [assignments, unTracks, unDets] = assign_detections(tracks, detB, AssignIoU);

    % Update matched tracks
    for a = 1:size(assignments,1)
        tIdx = assignments(a,1); dIdx = assignments(a,2);
        tracks(tIdx).bbox = detB(dIdx,:);
        tracks(tIdx).score = detS(dIdx);
        tracks(tIdx).age = tracks(tIdx).age + 1;
        tracks(tIdx).miss = 0;
        tracks(tIdx).frames = tracks(tIdx).frames + 1;
    end

    % Age unmatched tracks
    for t = reshape(unTracks,1,[])
        tracks(t).miss = tracks(t).miss + 1;
    end

    % Spawn new tracks
    for d = reshape(unDets,1,[])
        tracks(end+1) = struct('id', nextId, 'bbox', detB(d,:), 'age', 1, ...
            'miss', 0, 'frames', 1, 'score', detS(d)); %#ok<AGROW>
        nextId = nextId + 1;
    end

    % Prune stale tracks
    keepMask = [tracks.miss] <= MaxMissing;
    tracks = tracks(keepMask);

    J = overlay_tracks(I, tracks);
    writeVideo(vw,J);
end
close(vw);

% ---- SUMMARY ----
trackLengths = [tracks.frames];
if isempty(trackLengths)
    trackLengths = 0;
end
fprintf('Saved tracking video: %s | mean %.1f ms/frame (%.1f fps)\n', vidPath, mean(perFrameMs), 1000/mean(perFrameMs));
fprintf('Tracks created: %d | mean lifespan: %.1f frames\n', nextId-1, mean(trackLengths));
try, imwrite(J, fullfile(outFigureDir,'demo_track_frame.png')); end

fprintf('DONE.\n');

% ---- helpers ----
function ensure_dir(varargin)
for i=1:nargin, d=varargin{i}; if ~exist(d,'dir'), mkdir(d); end, end
end

function [assignments, unTracks, unDets] = assign_detections(tracks, dets, thr)
assignments = zeros(0,2);
unTracks = 1:numel(tracks);
unDets = 1:size(dets,1);
if isempty(tracks) || isempty(dets)
    return;
end

IoU = zeros(numel(tracks), size(dets,1));
for t = 1:numel(tracks)
    for d = 1:size(dets,1)
        IoU(t,d) = bbox_iou(tracks(t).bbox, dets(d,:));
    end
end

while true
    [val, idx] = max(IoU(:));
    if isempty(val) || val < thr
        break;
    end
    [tIdx, dIdx] = ind2sub(size(IoU), idx);
    assignments(end+1,:) = [tIdx, dIdx]; %#ok<AGROW>
    IoU(tIdx,:) = -Inf;
    IoU(:,dIdx) = -Inf;
end

if ~isempty(assignments)
    unTracks = setdiff(1:numel(tracks), assignments(:,1));
    unDets = setdiff(1:size(dets,1), assignments(:,2));
end
end

function v = bbox_iou(a, b)
ax1=a(1); ay1=a(2); ax2=a(1)+a(3); ay2=a(2)+a(4);
bx1=b(1); by1=b(2); bx2=b(1)+b(3); by2=b(2)+b(4);
interW = max(0, min(ax2,bx2) - max(ax1,bx1));
interH = max(0, min(ay2,by2) - max(ay1,by1));
inter = interW * interH;
union = a(3)*a(4) + b(3)*b(4) - inter;
v = inter / max(union, eps);
end

function J = overlay_tracks(I, tracks)
J = I;
if size(J,3)==1, J = repmat(J,[1 1 3]); end
for t = 1:numel(tracks)
    col = id_color(tracks(t).id);
    J = insertShape(J,'Rectangle',tracks(t).bbox,'LineWidth',3,'Color',col);
    label = sprintf('ID %d (%.2f)', tracks(t).id, tracks(t).score);
    J = insertText(J, tracks(t).bbox(1:2)+[0 -15], label, 'BoxOpacity',0.6, 'TextColor','white', 'FontSize',14, 'BoxColor',col);
end
end

function c = id_color(id)
% Generate a repeatable bright colour for each track id
palette = uint8(255 * hsv(20));
c = palette(mod(id-1,size(palette,1))+1, :);
end
end
