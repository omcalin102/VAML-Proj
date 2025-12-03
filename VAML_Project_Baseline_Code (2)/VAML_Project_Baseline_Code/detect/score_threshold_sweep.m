function T = score_threshold_sweep(I, model, varargin)
% Sweep MinScore and NMS IoU to see precision/recall trade-offs on one image with GT.

p = inputParser;
addParameter(p,'GTBoxes',[]);                              % ground truth boxes [N x 4] (changeable)
addParameter(p,'BaseWindow',[64 128]);                     % base window (changeable)
addParameter(p,'Step',8);                                  % stride px (changeable)
addParameter(p,'ScaleFactor',0.90);                        % pyramid factor (changeable)
addParameter(p,'MinScores',[-Inf 0 0.5 1 2]);              % MinScore grid (changeable)
addParameter(p,'NMS_IoUs',[0.3 0.4 0.5]);                  % NMS IoU grid (changeable)
addParameter(p,'IoU_TP',0.50);                             % TP IoU rule (changeable)
parse(p,varargin{:});
args = p.Results;

% Score all windows once (fast), then filter per threshold
[boxes, scores] = score_windows(I, model, ...
    'BaseWindow',args.BaseWindow, 'Step',args.Step, 'ScaleFactor',args.ScaleFactor, 'MinScore',-Inf); % prefilter (changeable)

rows = [];
for ms = args.MinScores
    keep = scores >= ms;
    B = boxes(keep,:); S = scores(keep);
    for nms = args.NMS_IoUs
        if isempty(B)
            prec=0; rec=0; f1=0; K=0;
        else
            [BB, SS] = nms(B, S, nms);                    % NMS IoU (changeable)
            if isempty(args.GTBoxes)
                prec=NaN; rec=NaN; f1=NaN; K=size(BB,1);
            else
                [tp,fp,fn] = match_detections(BB, args.GTBoxes, args.IoU_TP); % IoU_TP (changeable)
                prec = tp / max(1, tp+fp);
                rec  = tp / max(1, tp+fn);
                f1   = 2*prec*rec / max(1e-9,prec+rec);
                K    = size(BB,1);
            end
        end
        rows = [rows; ms, nms, K, prec, rec, f1]; %#ok<AGROW>
    end
end

T = array2table(rows,'VariableNames',{'MinScore','NMS_IoU','NumKeep','Precision','Recall','F1'});
end
