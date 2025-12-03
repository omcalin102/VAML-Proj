function GT = load_gt(gtFile, varargin)
% Returns struct with fields = image basenames, values = Nx4 [x y w h]

p = inputParser;
addParameter(p, 'DelimiterPattern', '[,\s]+');   % <-- change delimiter regex if needed
addParameter(p, 'HasHeader', false);             % <-- set true if first line is a header
addParameter(p, 'BoxFormat', 'xywh');            % <-- 'xywh' or 'xyxy'
parse(p, varargin{:});
args = p.Results;

assert(exist(gtFile,'file')==2, 'GT file not found: %s', gtFile);
lines = string(splitlines(fileread(gtFile)));
GT = struct();

rowStart = 1 + double(args.HasHeader);           % <-- skip header if present
for i = rowStart:numel(lines)
    L = strtrim(lines(i));
    if strlength(L)==0 || startsWith(L, "#"), continue; end   % skip blanks/comments
    toks = regexp(L, args.DelimiterPattern, 'split');
    if numel(toks) < 5, continue; end

    imgName = toks{1};
    nums = str2double(toks(2:end));
    nums = nums(~isnan(nums));

    % Support "name x y w h [x y w h ...]" or "name x1 y1 x2 y2 [...]"
    if strcmpi(args.BoxFormat,'xyxy')           % <-- switch if file is x1y1x2y2
        if mod(numel(nums),4)~=0, continue; end
        B = reshape(nums, 4, []).';
        % convert xyxy -> xywh
        B(:,3) = B(:,3) - B(:,1) + 1;
        B(:,4) = B(:,4) - B(:,2) + 1;
    else
        % Assume xywh; if an extra class/score column exists, drop it
        % e.g., name x y w h cls x y w h cls ...
        % Keep only multiples of 4 from the start
        n4 = floor(numel(nums)/4)*4;
        if n4 < 4, continue; end
        nums = nums(1:n4);
        B = reshape(nums, 4, []).';
    end

    key = strip_extension(imgName);
    if isfield(GT, key)
        GT.(key) = [GT.(key); B];
    else
        GT.(key) = B;
    end
end
end

function s = strip_extension(fname)
[~,s,~] = fileparts(fname);
end
