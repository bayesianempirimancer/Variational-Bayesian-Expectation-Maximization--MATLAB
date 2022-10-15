function [v,zeta_new] = normalize(u,zeta_old)
%% NORMALIZE: normalize rows of a matrix
% also returns normalization constants
u_sum = sum(u,1);
zeta_new = zeta_old .* u_sum;
v = bsxfun(@rdivide,u,u_sum);
end