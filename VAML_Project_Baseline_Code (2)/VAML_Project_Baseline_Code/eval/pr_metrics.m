function [acc, prec, rec, f1, cm] = pr_metrics(yhat, ytrue)
% Compute accuracy, precision, recall, F1 for binary labels.

% Map labels to {0,1}
ytrue = ytrue(:);
yhat  = yhat(:);
if any(ytrue==-1) || any(yhat==-1)
    ytrue = (ytrue==1);                      % treat +1 as positive (changeable)
    yhat  = (yhat==1);                       % treat +1 as positive (changeable)
else
    ytrue = logical(ytrue);
    yhat  = logical(yhat);
end

tp = sum(yhat & ytrue);
fp = sum(yhat & ~ytrue);
fn = sum(~yhat & ytrue);
tn = sum(~yhat & ~ytrue);

acc  = (tp+tn) / max(1, tp+fp+fn+tn);
prec = tp       / max(1, tp+fp);             % precision definition (changeable)
rec  = tp       / max(1, tp+fn);             % recall definition (changeable)
f1   = 2*prec*rec / max(1e-12, prec+rec);    % F1 epsilon (changeable)

cm = [tp fp; fn tn];                         % confusion matrix layout (changeable)
end
