function progress_bar(k, n, t0, label)
%PROGRESS_BAR Simple textual progress with ETA and elapsed time.
%   progress_bar(k, n, t0, label) prints a carriage-return updated line.
%   t0 is a tic/toc handle; label is optional prefix text.

if nargin < 4, label = ''; end
if nargin < 3 || isempty(t0), t0 = tic; end

elapsed = toc(t0);
eta = (elapsed / max(1,k)) * (n - k);
frac = k / max(1,n);

msg = sprintf('\r%s %4d/%4d (%.0f%%) | elapsed %.1fs | ETA %.1fs', ...
    label, k, n, round(100*frac), elapsed, max(0,eta));
fprintf('%s', msg);
if k == n
    fprintf('\n');
end
end
