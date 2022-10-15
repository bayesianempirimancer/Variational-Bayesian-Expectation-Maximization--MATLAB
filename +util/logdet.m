function [res,R] = logdet(A)  % R'*R = A
    R = chol(A);
	res = 2*sum(log(diag(R)));
end


