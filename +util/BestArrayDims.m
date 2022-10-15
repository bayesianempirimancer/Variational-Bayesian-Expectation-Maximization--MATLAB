function [nr nc] = BestArrayDims(n,wv)
% Best number of rows and columns for arrays (or subplots), weighted by user
%
% [nr nc] = BestArrayDims(n) returns an estimate of the best number of 
% rows and columns to use in your cell arrays or subplots if you do not
% know in advance how many elemnts there will be in advance. The number of
% suggested rows and columns are returned in nr and nc, respectively.
%
% EXAMPLE:
%   % what is the best subplot arrangement for 17 plots?
%   >> [nr nc] = BestArrayDims(17)
% nr =
%      6
% nc =
%      3
%
% [nr nc] = BestArrayDims(n,wv) The algorithm weights the amount of
% non-wasted space against the squareness of the resulting configuration
% equally by default (wv=0.5). To use your own weightings use the wv input.
% A larger value of wv gives preference to square configurations and
% smaller value gives preference to rectangular configurations that avoid
% empty cells. wv must be between 0 and 1.
%
% EXAMPLE
%   % the best array dimensions for 14 elements
%   >> [nr nc] = BestArrayDims(14)
% nr =
%      5
% nc =
%      3
%   % what if I prefer square arrangements over rectangles?
%   >> [nr nc] = BestArrayDims(14,0.6)
% nr =
%      4
% nc =
%      4
%   % what if I greatly prefer rectangles (less empty space)?
%   >> [nr nc] = BestArrayDims(14,0.2)
% nr =
%      7
% nc =
%      2
%   % what if I want to enforce a single column of subplots?
%   >> [nr nc] = BestArrayDims(14,0)
% nr =
%     14
% nc =
%      1
%
%
% %%% ZCD %%%
%

% the side-length of the largest adequit square layout
dl = ceil(sqrt(n));

% check inputs
if nargin==1
    wv = 0.5;     % equal weightings is default
else
    if numel(wv)~=1 || ~isreal(wv) || wv>1 || wv<0
        error('Weight must be a real scalar between 0 and 1')
    else
        % don't bother with the code, just enforce a square
        if wv==1, nr = dl; nc = dl; return; end
        % don't bother with the code, just enforce a column
        if wv==0, nr = n; nc = 1; return; end
    end
end
    
% all possible and non-excessive widht/height combinations
h = ceil(n./(1:dl));
w = 1:dl;
% calculate wasted cells & normalize for comparison to squareness
waste = (w.*h-n)/n;
% surface area to volume ratio
savr = (2*w+2*h)./(w.*h);
% the final weighted cost function
cost = wv*savr+(1-wv)*waste;

% minimize cost (note this index is also the subplot width)
[v nc] = min(cost);
% use the width to index the subplot height
nr = h(nc);

% debug
% [h;w;waste*n;savr;cost]



