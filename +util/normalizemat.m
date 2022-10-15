function [v,zeta_new] = normalizemat(u,zeta_old)
%% NORMALIZE: normalize rows of a matrix
% also returns normalization constants
u_sum = sum(sum(abs(u)));
zeta_new = zeta_old .* u_sum;
v = u / u_sum;
end