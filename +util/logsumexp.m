function s = logsumexp(a)
if size(a,2) < 2
    s = a;
else
    y = max(a,[],2);
    s = bsxfun(@minus,a,y);
    s = y + log(sum(exp(s),2));
    %s(~isfinite(y)) = y(~isfinite(y));
end
end