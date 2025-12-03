function M = parse_dataset_qub(file)
% Robust parser for the course's test.dataset style.
% Map key = normalized filename (via make_key/variants) -> Nx4 [x y w h]
assert(exist(file,'file')==2, 'Dataset not found: %s', file);

txt = fileread(file);
L = regexp(txt,'\r\n|\n','split'); L = L(:);

M = containers.Map('KeyType','char','ValueType','any');
i = 1;
while i <= numel(L)
    line = strtrim(L{i});
    if line==""; i = i+1; continue; end

    toks = split(line);
    rawImg = char(toks{1});
    key    = make_key(rawImg);     % ← normalize

    rest = str2double(toks(2:end));
    boxes = [];

    if numel(rest)==4 && all(~isnan(rest))
        boxes = rest(:)';                 % one box same line
        i = i+1;
    elseif numel(rest)==1 && ~isnan(rest) && rest>=0
        N = round(rest);
        i = i+1;
        for j = 1:N
            if i>numel(L), break; end
            xywh = str2num(L{i}); %#ok<ST2NM>
            if numel(xywh)==4, boxes = [boxes; xywh(:)']; end %#ok<AGROW>
            i = i+1;
        end
    else
        % fallback: next line one box
        if i+1 <= numel(L)
            xywh = str2num(L{i+1}); %#ok<ST2NM>
            if numel(xywh)==4
                boxes = xywh(:)';
                i = i+2;
            else
                i = i+1;
            end
        else
            i = i+1;
        end
    end

    if isempty(boxes), boxes = zeros(0,4); end
    M(key) = boxes;
end
end
