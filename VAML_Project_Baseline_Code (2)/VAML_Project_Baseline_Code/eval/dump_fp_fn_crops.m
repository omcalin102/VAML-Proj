function dump_fp_fn_crops(imgDir, gtFile, modelPath, outDir, varargin)
% Saves crops for FP detections and FN ground-truth boxes to visually inspect failures.

p = inputParser;
addParameter(p,'BaseWindow',[64 128]);
addParameter(p,'Step',16);
addParameter(p,'ResizeLongSide',480);
addParameter(p,'MinScore',0.6);
addParameter(p,'NMS_IoU',0.25);
addParameter(p,'IoU_TP',0.5);
parse(p,varargin{:}); a=p.Results;

if ~exist(outDir,'dir'), mkdir(outDir); end
S = load(modelPath); model = S.model;
GT = parse_dataset_qub(gtFile);

imgs = [dir(fullfile(imgDir,'*.jpg')); dir(fullfile(imgDir,'*.png')); ...
        dir(fullfile(imgDir,'*.JPG')); dir(fullfile(imgDir,'*.PNG'))];
[~,ord] = sort({imgs.name}); imgs = imgs(ord);

for k=1:numel(imgs)
    name=imgs(k).name; inpth=fullfile(imgs(k).folder,name);
    I0=imread(inpth);
    key=lower(erase(name,{'.jpg','.png','.JPG','.PNG'}));
    if ~isKey(GT,key), continue; end
    G = GT(key);

    % detect
    [H,W,~]=size(I0); longSide=max(H,W);
    scale=1.0; I=I0; step=a.Step;
    if a.ResizeLongSide>0 && longSide>a.ResizeLongSide
        scale=a.ResizeLongSide/double(longSide);
        I=imresize(I0,scale); step=max(4,round(a.Step*scale));
    end
    [B,S]=score_windows(I,model,'BaseWindow',a.BaseWindow,'Step',step, ...
                        'MinScore',a.MinScore,'MaxWindows',200,'Verbose',false);
    [B,S]=nms_fast(B,S,a.NMS_IoU);
    if scale~=1.0, B=round(B./scale); end

    % match
    used=false(size(G,1),1);
    for i=1:size(B,1)
        [mx,ix]=max(iou(B(i,:),G));
        if isempty(mx) || mx<a.IoU_TP || used(ix)
            crop_and_save(I0,B(i,:), fullfile(outDir,sprintf('%s_FP_%02d.png',key,i)));
        else
            used(ix)=true;
        end
    end
    % any remaining GT are FN
    miss = find(~used);
    for j=1:numel(miss)
        crop_and_save(I0,G(miss(j),:), fullfile(outDir,sprintf('%s_FN_%02d.png',key,j)));
    end
end
fprintf('Saved FP/FN crops → %s\n', outDir);
end

% helpers
function crop_and_save(I,box,out)
x=max(1,box(1)); y=max(1,box(2)); w=max(1,box(3)); h=max(1,box(4));
x2=min(size(I,2),x+w-1); y2=min(size(I,1),y+h-1);
C=I(y:y2,x:x2,:); imwrite(C,out);
end
function v = iou(b,B)
if isempty(B), v=[]; return; end
x1=B(:,1); y1=B(:,2); x2=x1+B(:,3)-1; y2=y1+B(:,4)-1;
xx1=max(b(1),x1); yy1=max(b(2),y1);
xx2=min(b(1)+b(3)-1,x2); yy2=min(b(2)+b(4)-1,y2);
w=max(0,xx2-xx1+1); h=max(0,yy2-yy1+1);
inter=w.*h; area1=b(3)*b(4); area2=(x2-x1+1).*(y2-y1+1);
v = inter./max(1e-9, area1+area2-inter);
end
