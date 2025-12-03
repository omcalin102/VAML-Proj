function check_dataset_alignment(imgDir, gtFile, outTxt)
% Verifies test.dataset filenames match images in imgDir (case/padding tolerant).
% Writes both a TXT summary and a CSV audit with row: Image, FoundGT, UsedGTKey, NumGT

if nargin<3
    outTxt = fullfile('results','metrics','run5','dataset_alignment.txt');
end
outCsv = fullfile(fileparts(outTxt), 'dataset_alignment.csv');
if ~exist(fileparts(outTxt),'dir'), mkdir(fileparts(outTxt)); end

imgs = [dir(fullfile(imgDir,'*.jpg')); dir(fullfile(imgDir,'*.png')); ...
        dir(fullfile(imgDir,'*.JPG')); dir(fullfile(imgDir,'*.PNG'))];

GT = parse_dataset_qub(gtFile);

missingGT = {};
auditRows = {};

for k=1:numel(imgs)
    name = imgs(k).name;
    [has, boxes, usedKey] = gt_lookup(GT, name);
    if ~has, missingGT{end+1} = name; end %#ok<AGROW>
    auditRows(end+1,:) = {name, has, usedKey, size(boxes,1)}; %#ok<AGROW>
end

fid = fopen(outTxt,'w');
fprintf(fid, "Images in folder: %d\nGT entries: %d\n\n", numel(imgs), GT.Count);
fprintf(fid, "Images with NO GT: %d\n", numel(missingGT));
for i=1:numel(missingGT), fprintf(fid, "  %s\n", missingGT{i}); end
fclose(fid);

T = cell2table(auditRows, 'VariableNames', {'Image','FoundGT','UsedGTKey','NumGT'});
writetable(T, outCsv);

fprintf('Alignment report → %s\nAudit CSV         → %s\n', outTxt, outCsv);
end
