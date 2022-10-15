function [s,z] = lognormalize(A,dim)
%LOGNORMALIZE: normalize the matrix A along dimension dim

% find log of normalizing constant
if dim == 1
    z = util.logsumexp(A')';
else
    z = util.logsumexp(A);
end

% normalize by subtracting off constant
s = bsxfun(@minus,A,z);

end