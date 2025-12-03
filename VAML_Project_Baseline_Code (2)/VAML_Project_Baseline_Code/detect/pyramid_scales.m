function scales = pyramid_scales(imgHW, baseWindow, scaleFactor, varargin)
% scaleFactor: 0.85–0.95 typical  % <-- tunable

p = inputParser;
addParameter(p, 'MinShortSide', min(baseWindow));  % min pyramid size stop
addParameter(p, 'Start', 1.0);                     % starting scale
addParameter(p, 'MaxLevels', inf);                 % cap number of levels
parse(p, varargin{:});
args = p.Results;

H = imgHW(1); W = imgHW(2);
bw = baseWindow(1); bh = baseWindow(2);

assert(scaleFactor > 0 && scaleFactor < 1, 'scaleFactor must be in (0,1)');

scales = args.Start;                                % start scale (changeable)
level  = 1;

fits        = @(s) (H*s >= bh) && (W*s >= bw);      % window-fit rule (can alter)
shortSideOK = @(s) min(H,W)*s >= args.MinShortSide; % short-side stop (changeable)

while fits(scales(end)) && shortSideOK(scales(end)) && level < args.MaxLevels  % MaxLevels cap
    next = scales(end) * scaleFactor;               % scale step factor (changeable)
    if ~fits(next) || ~shortSideOK(next), break; end
    scales(end+1) = next; %#ok<AGROW>
    level = level + 1;
end

scales = unique(scales, 'stable');                  % keep order/uniqueness
end
