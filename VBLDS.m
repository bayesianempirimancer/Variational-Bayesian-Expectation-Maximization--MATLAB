classdef VBLDS < handle
    % Bayesian linear dynamical system
    
    % x_t+1 ~ N(x_t+1 | Ax_t, Q)
    % y_t ~ N(y_t | Cx_t, R)
    % x_0 ~ N(x_0 | mu0, sigma0)    
    
    % parms = {A, C, Q, R}
    
    % p(A, Q) = matrix normal wishart
    % p(C, R) = matrix normal wishart
    
    % inference done via VBEM, referenced from Beal 2003         
    
    properties
        
        T % num trials
        t % trial lengths
        iters % iters spent fitting
        obs % data
        
        % Parameters
        k % dim of state space
        d % dim of obs. space
                
        A % state dynamics matrix + state noise
        C % emission matrix + obs. noise
        
        mu0 % auxiliary zeroth state mean
        invSigma0 % auxiliary zeroth state precision     
        
        % ELBO
        
        L
        dLs
        Ls
        lnpY % marginal loglike
        
        % Latents
        % indexing is a major pain here bc of auxiliary state
        % see notes throughout
        % to get alpha_t, access alpha(t)
        % to get beta_t, access beta(t+1)
        % to get X_t, access X(t+1)
        
        X_means % X{trial} = [k, time]
        X_covars % X{trial} = [k, k, time]
        X_crossts % X{trial} = [k, k, time] -- covars on p(x_t, x_t+1 | Y)     
        
        % filtered dists. on x
        alpha_means 
        alpha_Ps
        
        % p(x_t | y_t+1...y_T)
        beta_means 
        beta_Ps
        
        % for debugging, these store the values at each iteration
        lnpYs
        AKLs
        CKLs
        R2s
        
        % transformed Chat
        Ctrue
        
    end
    
    methods
        
        function self = VBLDS(k, d)
            % constructor
            
            self.L = -inf;
            self.iters = 0;
            
            self.k = k;
            self.d = d;
                        
            self.A = dists.expfam.matrixnormalWishart(...
                zeros(k,k),...
                eye(k),... 
                eye(k)); 
                                
            self.C = dists.expfam.matrixnormalWishart(...
                zeros(d,k),...
                eye(d),... 
                eye(k));
                                    
            self.mu0 = zeros(k,1);
            self.invSigma0 = 1e-7 * eye(k);                        
            
        end
        
        function fit(self, Y, iters)
            % fit function
            
            self.T = size(Y,1);
            for trial = 1:self.T
                self.t(trial) = size(Y{trial},2);
            end
            self.obs = Y;
            
            i = 1;
            while (i <= iters)
                                
                updateLatents(self,Y);                
                updateELBO(self, self.iters + i);                 
                updateParms(self,Y);
                
                if mod(i, iters/10) == 0
                    fprintf('iters done: %d\n', i);
                end
                
                % UNCOMMENT this to see R2s over iters
%                 self.R2s(:,self.iters + i) = self.R2;
       
                i = i + 1;                              
            end
            
            self.iters = self.iters + i - 1;
                        
        end
        
        function updateLatents(self, Y)
           % NOTE there are some redundant matrix computations here 
            
           % E step
           % self.alpha are filtered state dists.
           % self.beta are p(y_t+1...y_T | x_t)
           % self.X are smoothed state dists.           
                     
           self.lnpY = 0; % for calculation of F / ELBO
           
           % parameter expectations
           ATinvQ = self.A.EXTinvU;
           CTinvR = self.C.EXTinvU;
           ATinvQA = self.A.EXTinvUX;
           CTinvRC = self.C.EXTinvUX;
           invQ = self.A.EinvU;
           invR = self.C.EinvU;           
                                 
           for trial = 1:self.T
               
               n = self.t(trial);
               
               % need these for both halves
               % to get sigmaStar_t, access SigmaStars(t+1)
               invSigmaStars = zeros(self.k, self.k, n); 
               
               % Forward
               for time = 1:n                                                     
                   
                   y_t = Y{trial}(:,time);
                   
                   % initial step relies on auxiliary x0
                   % invSigmaStar is always from previous step / init.
                   if time == 1
                       mu_prev = self.mu0;
                       invSigma_prev = self.invSigma0;                      
                       invSigmaStar = self.invSigma0 + ATinvQA;
                       invSigmaStars(:,:,time) = invSigmaStar;
                   else % recursion
                       invSigma_prev = invSigma_t;       
                       mu_prev = mu_t;                        
                       invSigmaStar = invSigma_t + ATinvQA;
                       invSigmaStars(:,:,time) = invSigmaStar;                          
                   end                                    
                   
                   % alphas
                   invSigma_t = invQ + CTinvRC - ATinvQ' / invSigmaStar * ATinvQ;
                   self.alpha_Ps{trial}(:,:,time) = invSigma_t;
                   
                   mu_t = invSigma_t \ (CTinvR * y_t + ATinvQ' / invSigmaStar * invSigma_prev * mu_prev);  
                   self.alpha_means{trial}(:,time) = mu_t;                   
                                               
                   % E log like of y_time | y_1...y_time-1    
                   % does ElogdetinvU work?
                   loglike = self.d * log(2*pi) - self.C.ElogdetinvU - log(det(invSigma_prev / invSigmaStar / invSigma_t)) ...
                       + mu_prev' * invSigma_prev * mu_prev - mu_t' * invSigma_t * mu_t + y_t' * invR * y_t ...
                       - (invSigma_prev * mu_prev)' / invSigmaStar * invSigma_prev * mu_prev;
                   loglike = -1/2 * loglike;                   
                   self.lnpY = self.lnpY + loglike;
                                       
               end
               
               % Backward
               % beta: to get t, access t+1
               % note that crosst covars are calculated here                              
               % crosst: to get cross_t,t+1 access t+1
               
               % init
               invPsi_t = zeros(self.k, self.k);               
               self.beta_Ps{trial}(:,:,n+1) = invPsi_t;

               eta_t = zeros(self.k,1);
               self.beta_means{trial}(:,n+1) = eta_t;                                       
                                            
               for time = fliplr(1:n)
                   
                   y = Y{trial}(:,time);
                   
                   invPsiStar = invQ + CTinvRC + invPsi_t;               
                   invPsi_after = invPsi_t;
                   eta_after = eta_t;                           
                   
                   % note invPsiStar is always from previous step (later in
                   % time)

                   % betas
                   invPsi_t = ATinvQA - ATinvQ / invPsiStar * ATinvQ';
                   self.beta_Ps{trial}(:,:,time) = invPsi_t;
                   
                   eta_t = invPsi_t \ ATinvQ / invPsiStar * (CTinvR * y + invPsi_after * eta_after);
                   self.beta_means{trial}(:,time) = eta_t;                      
                   
                   % crosst covars
                   % i think this is wrong because we want <(x_t+1 - omega_t+1) * (x_t - omega_t)'>
                   % except whatever tweaks i tried made things worse
                   invSigmaStar = invSigmaStars(:,:,time);                   
                   invPart = (invQ + CTinvRC + invPsi_after - ATinvQ' / invSigmaStar * ATinvQ);
                   upsilon_crosst = invSigmaStar \ ATinvQ / invPart;
                   self.X_crossts{trial}(:,:,time) = upsilon_crosst;                              
                   
               end                       
               
               % combine alphas and betas
               % X: to get t, access t+1
               
               for time=0:n          
                   
                   % alpha message
                   if time == 0
                       mu_t = self.mu0;
                       invSigma_t = self.invSigma0;
                   else
                       mu_t = self.alpha_means{trial}(:,time);
                       invSigma_t = self.alpha_Ps{trial}(:,:,time);
                   end                                 
                                                       
                   % beta message
                   eta_t = self.beta_means{trial}(:,time+1);
                   invPsi_t = self.beta_Ps{trial}(:,:,time+1);                

                   % combine                   
                   invUpsilon_t = invSigma_t + invPsi_t;
                   upsilon_t = inv(invUpsilon_t);
                   self.X_covars{trial}(:,:,time+1) = upsilon_t;
                   
                   omega_t = invUpsilon_t \ (invSigma_t * mu_t + invPsi_t * eta_t);
                   self.X_means{trial}(:,time+1) = omega_t;    
                   
               end               
               
           end           
                      
        end
        
        function updateELBO(self, i)
            % update lower bound
            
            self.lnpYs(i) = self.lnpY;
            self.AKLs(i) = self.A.KLqprior;
            self.CKLs(i) = self.C.KLqprior;
                        
            newL = - self.A.KLqprior - self.C.KLqprior + self.lnpY;
            if i > 1
                self.dLs(i-1) = newL - self.L;
            end
            self.L = newL;
            self.Ls(i) = self.L;        
            
        end
        
        function updateParms(self, Y)
           % M step
           
           XXnoAux = zeros(self.k,self.k);
           XXnoEnds = zeros(self.k,self.k);
           XXcrosst = zeros(self.k,self.k);
           YX = zeros(self.d,self.k);
           YY = zeros(self.d,self.d);
                      
           for trial = 1:self.T               
                              
               for time = 1:self.t(trial)
                   % this works because there are t crossts and t+1 x's
                   y_t = Y{trial}(:,time);
                   omega_t = self.X_means{trial}(:,time);
                   omega_t1 = self.X_means{trial}(:,time+1);
                   upsilon_t = self.X_covars{trial}(:,:,time);
                   upsilon_t1 = self.X_covars{trial}(:,:,time+1);                   
                   upsilon_crosst = self.X_crossts{trial}(:,:,time);                   
                   
                   XXnoEnds = XXnoEnds + upsilon_t + omega_t * omega_t';       
                   XXcrosst = XXcrosst + upsilon_crosst + omega_t1 * omega_t'; 
                   XXnoAux = XXnoAux + upsilon_t1 + omega_t1 * omega_t1';
                   YX = YX + y_t * omega_t1';
                   YY = YY + y_t * y_t';
                                      
               end                              
               
           end                   
                                                        
           % update           
           N = sum(self.t);
           self.A.updateSS(XXnoEnds/N, XXcrosst/N, XXnoAux/N, N);
           N = sum(self.t);
           self.C.updateSS(XXnoAux/N, YX/N, YY/N,N);                    
           
           newMu0 = zeros(self.k,1);                                
           for trial = 1:self.T
               newMu0 = newMu0 + self.X_means{trial}(:,1);
           end
           self.mu0 = newMu0 / self.T;
           
           newInvSigma0 = zeros(self.k,self.k);           
           for trial = 1:self.T
               dif = self.mu0 - self.X_means{trial}(:,1);
               newInvSigma0 = newInvSigma0 + self.X_covars{trial}(:,:,1) + dif * dif';
           end                                
           self.invSigma0 = newInvSigma0 / self.T;                        
           
        end
        
        function [pred_means, pred_covars] = getPreds(self)
           % returns mean and covar of p(y_t | data) for all t
           pred_means = cell(self.T,1);
           pred_covars = cell(self.T,1);                      
                      
           Cmean = self.C.mean;
           R = inv(self.C.invU.mean);
                      
           for trial = 1:self.T
              for time = 1:self.t(trial)
                 omega_t = self.X_means{trial}(:,time+1);
                 upsilon_t = self.X_covars{trial}(:,:,time+1);
                 CUCT = self.C.EXAXT(upsilon_t);
                 
                 pred_means{trial}(:,time) = Cmean * omega_t;
                 pred_covars{trial}(:,:,time) = CUCT + R; % is this right?                 
              end
           end
           
        end
        
        function r = r(self,Xtrue)
            % Xtrue should by t x k
            % calcs r from canoncorr and Ctrue
            
            Xhat = zeros(sum(self.t),self.k);
            for trial = 1:self.T
                for time = 1:self.t(trial)
                    omega_t = self.X_means{trial}(:,time+1);
                    if trial == 1
                        index = time;
                    else
                        index = sum(self.t(1:trial-1)) + time;
                    end                    
                    Xhat(index,:) = omega_t';
                end
            end   
            [a,b,r] = canoncorr(Xtrue, Xhat);                  
            self.Ctrue = self.C.mean * (b \ a)';
            
        end        
        
        function R2 = R2(self)
            % compute R2 for each observed dimension
            SSR = zeros(self.d,1);
            var = zeros(self.d,1);
            Ymean = mean([self.obs{:}],2);
            Cmean = self.C.mean;
            
            for trial = 1:self.T                
                for time = 1:self.t(trial)                    
                    x = self.X_means{trial}(:,time+1);
                    y = self.obs{trial}(:,time);
                    pred = Cmean*x;       
                    for dim = 1:self.d
                        SSR(dim) = SSR(dim) + (y(dim) - pred(dim))^2;
                        var(dim) = var(dim) + (y(dim) - Ymean(dim))^2;
                    end
                end
            end
            
            R2 = 1 - SSR./var;
            
        end
        
        function plotLs(self)
           % plots L over iterations           
           plot(self.Ls(3:end))
           title("L by iteration")
        end   
        
        function plotR2s(self)
            for i = 1:self.d
               figure()
               plot(self.R2s(i,:));
               title("R2 for dim " + i);
            end
        end
        
        function negdLs(self)
           % returns the negative changes in L
           idx = self.dLs < 0;
           self.dLs(idx)
        end        
                
    end    
    
end

% helpers
function bool = isPSD(V)
    bool = all(all(V == V')) && all(eig(V) > eps);
end

function res = makePSD(V)
   res = V;
   if ~(isPSD(V))
      [P,D] = eig(V);
      D = diag(D);                      
      D(D<eps) = eps;                      
      D = diag(D);
      res = P * D * P';
   end    
end

