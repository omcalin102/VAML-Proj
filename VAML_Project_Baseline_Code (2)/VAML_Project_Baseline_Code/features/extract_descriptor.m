function feat = extract_descriptor(I, descCfg)
%EXTRACT_DESCRIPTOR Unified feature wrapper (HOG / pixels + optional PCA).
%   feat = EXTRACT_DESCRIPTOR(I, descCfg) computes a row feature vector for
%   image patch I according to descCfg.Type ('hog' or 'pixels'). When
%   descCfg.PCA is non-empty, the raw descriptor is projected using the
%   stored PCA transform so detection uses the exact same embedding as
%   training.

arguments
    I
    descCfg struct
end

descType = lower(string(descCfg.Type));

switch descType
    case "hog"
        feat = extract_hog(I, ...
            'ResizeTo', descCfg.ResizeTo, ...
            'CellSize', descCfg.CellSize, ...
            'BlockSize', descCfg.BlockSize, ...
            'BlockOverlap', descCfg.BlockOverlap, ...
            'NumBins', descCfg.NumBins);
    case "hogparts"
        feat = extract_hog_parts(I, ...
            'ResizeTo', descCfg.ResizeTo, ...
            'CellSize', descCfg.CellSize, ...
            'BlockSize', descCfg.BlockSize, ...
            'BlockOverlap', descCfg.BlockOverlap, ...
            'NumBins', descCfg.NumBins);
    case "pixels"
        feat = extract_pixels(I, 'ResizeTo', descCfg.ResizeTo);
    otherwise
        error('Unsupported descriptor type: %s', descType);
end

% Optional PCA projection
if isfield(descCfg, 'PCA') && ~isempty(descCfg.PCA)
    mu = descCfg.PCA.Mu(:)';
    coeff = descCfg.PCA.Coeff;
    feat = single((double(feat) - double(mu)) * double(coeff));
end

if iscolumn(feat), feat = feat'; end
feat = single(feat);
end
