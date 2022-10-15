classdef HMM < handle
    
    % Discrete hidden markov model with Exponential Family observations.
    % See GPEF for the specification of observation types.  
    % Algorithm uses standard forward backward agorithm applied to
    % geomemtric mean of the transition probability matrix to exactly
    % integrate out latent states via enumeration.  
    
    % For large state spaces with known sparsity structure set sparse = 1 to encode the transition
    % probability matrix in sparse format and place zeros on the Transition probability prior Aalpha_0 
    % as the algorithm enforces the constraint Aalpha_0(i,j) == 0 => <A(i,j)> = 0 
    
    % Usage HMM.update(data,iters) expects data{i}(j,t) to be the
    % obserrvation of dimension j at time t for time series i.  There is no
    % need for time series to be of the same length.
    
    % It its current form this algorithm is quite slow.  Future work should
    % include parallelizing forward backward updates using masked batches
    % of time series.
    
    % Latent stat assignments are stored in p{i}(j,t) which gives the
    % probability that the system is in state j and time t in time series
    % i.
    
    properties
        dim % dimension of the state space
        D % dimension of the observation space
        pi0 %initial distribiutions
        A % transition probabilities
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  
        obsModels 
        p % DxT state assignment probability
        Exx
        logZ
        L
        sparse
    end

    methods
        function self = HMM(dim,D,obsTypes,Aalpha_0,pi0alpha_0,A,pi0)
            self.dim = dim;
            self.D = D;
            if(size(Aalpha_0)~=[dim,dim])
                Aalpha_0=ones(dim)*Aalpha_0(1);
            end
            sparse = issparse(Aalpha_0);            
            if(size(pi0alpha_0)~=[dim,1])
                pi0alpha_0=pi0alpha_0(1)*ones(dim,1);
            end
            if(~exist('A','var')) 
                self.A=dists.transition(dim,Aalpha_0);
            else
                self.A=dists.transition(dim,Aalpha_0,A);;
            end
            if(~exist('pi0','var'))
                self.pi0=dists.expfam.dirichlet(dim,pi0alpha_0);
            else
                self.pi0=dists.expfam.dirichlet(dim,pi0alpha_0,pi0);
            end
            
            if(ischar(obsTypes))
                fprintf('using empty observation distribution.\n');
                self.obsTypes=obsTypes;
                self.obsModels=obsTypes;
            elseif(isempty(obsTypes))
                fprintf('Defaulting to normally distributed observations.\n');
                self.obsTypes{1}.dist = 'mvn';
                self.obsTypes{1}.idx = [1:D];
                for i=1:dim
                    self.obsModels{i}=dists.GPEF(self.obsTypes);
                end
            else
                self.obsTypes = obsTypes;
                for i=1:dim
                    self.obsModels{i}=dists.GPEF(self.obsTypes);
                end
            end
            
        end
        
        function res = obsloglike(self,data)
            % outputs a cell array of  T x dim matrix of likelihoods 
            % assuming standard data of the form data{trials}(D , T)
            res={};
            for i=1:numel(data)
                res{i}=zeros(self.dim,size(data{i},2));
                for k=1:self.dim
                    res{i}(k,:) = self.obsModels{k}.Eloglikelihood(data{i}')';  
                    % these primes needs to be fixed in Eloglikelihood computation
                end
            end
        end
        
        function L = update(self,data,iters)
            if(~exist('iters','var'))
                iters=1;
            end         
            for i=1:iters
                [SExx,SEx0] = self.update_states(data);
                L = sum(cellfun(@sum,self.logZ)) - self.KLqprior;
                self.L = L;
                self.A.update(SExx);
                self.pi0.update(SEx0);
                datacat = [data{1:end}];
                pcat = [self.p{1:end}];
                for k=1:self.dim
                    self.obsModels{k}.update(datacat',pcat(k,:)');
                end
            end
        end
        
        function updateparms(self,data,p) 
            % assumes states are up to date so that p, Exx, and Ex0 are too.
            % Also assumes that p is a vector of length numel(data);
            % the expected use here is to call self.Eloglikelihood to
            % compute trial cluster assignments then to update parameters 
            % according to according to
            if(~exist('p','var'))
                n=numel(data);
                p=ones(numel(data),1); 
            else
                n=sum(p);
            end
            if(isempty(data))
                for k=1:self.dim
                    self.obsModels{k}.update({},0);
                end
                return
            end
            if(issparse(self.A.alpha))
                SExx=sparse(self.dim,self.dim);
            else
                SExx = zeros(self.dim,self.dim);
            end
            SEx0 = zeros(self.dim,1);
            for i=1:numel(data);
                SExx = SExx + self.Exx{i}*p(i);
                SEx0 = SEx0 + self.p{i}(:,1)*p(i);
                pcat{i} = self.p{i}*p(i);
            end            
            self.A.update(SExx);
            self.pi0.update(SEx0);
            datacat = [data{1:end}];
            pcat = [pcat{1:end}];
            for k=1:self.dim
                self.obsModels{k}.update(datacat',pcat(k,:)');
            end
        end
        
        function L = updateMarkovparms(self,data,p,didx) 
            % assumes states are up to date so that p, Exx, and Ex0 are too.
            % Also assumes that p is a vector of length numel(data);
            % the expected use here is to call self.Eloglikelihood to
            % compute trial cluster assignments then to update parameters 
            % according to according to
            if(~exist('p','var'))
                n=numel(data);
                p=ones(numel(data),1); 
            elseif(isempty(p))
                n=numel(data);
                p=ones(numel(data),1);                 
            else
                n=sum(p);
            end
            if(~exist('didx','var'))
                didx=[1:numel(data)];
            end
            SExx = sparse(self.dim,self.dim);
            SEx0 = zeros(self.dim,1);
            for i=didx
                SExx = SExx + self.Exx{i}*p(i);
                SEx0 = SEx0 + self.p{i}(:,1)*p(i);
            end 
            self.A.update(SExx);
            self.pi0.update(SEx0);
        end
                
        function res = Eloglikelihood(self,data)
            if(~exist('data','var'))
                res = self.logZ;
            else
                self.update_states(data);
                res = self.logZ;
            end
        end

        function res = Ep(self,data)
            if(~exist('data','var'))
                res = self.p;
            else
                for i=1:numel(data) % can be parallelized 
                    obslike = self.obsloglike(data);
                    [p,Exx,logZ] = self.forwardbackward(obslike{i},self.A.loggeomean,self.pi0.loggeomean);
                    self.p{i} = p;
                    self.Exx{i} = Exx;
                    self.logZ{i} = logZ;               
                end
                res = self.p;
            end
        end
        
        function [SExx,SEx0] = update_states(self,data,obslike,didx) 
            % note that data is again assumed to be a cell array
            % This routine only returns state probabilities and sufficient
            % statistics for A and pi.
            if(~exist('obslike','var'))
                obslike = self.obsloglike(data);
            end
            if(~exist('didx','var'))
                didx=[1:numel(data)];
            end
            if(issparse(self.A.alpha))
                SExx = sparse(self.dim,self.dim);
            else
                SExx = zeros(self.dim,self.dim);
            end
            SEx0 = zeros(self.dim,1);   
            for i=didx % can be parallelized 
                [p,Exx,logZ] = self.forwardbackward(obslike{i},self.A.loggeomean,self.pi0.loggeomean);
                self.p{i} = p;
                self.Exx{i} = Exx;
                self.logZ{i} = logZ;               
                SExx = SExx + Exx;
                SEx0 = SEx0 + p(:,1);
            end
            self.p = self.p(1:length(data));
            self.Exx = self.Exx(1:length(data));
            self.logZ = self.logZ(1:length(data));
        end
        
        function [p,SExx,logZ,xi] = forwardbackward(self,loglike,logA,logpi) 
            if(isempty(loglike))
                p=[];
                SExx=[];
                logZ=[];
                xi=[];
                return
            end
            % forward propogation
            T=size(loglike,2);
            A=spfun(@exp,logA);
%            A=sparse(exp(logA)); % comment out for non-sparse A computation;
            pi=exp(logpi);
%            [loglike,logz] = util.lognormalize(loglike,1);
            
            if(size(loglike,1)>1)
                logz=max(loglike);
            else
                logz=ones(1,size(loglike,2))*max(loglike);
            end
            loglike=bsxfun(@plus,loglike,-logz);
                        
            like = exp(loglike);
%            z = exp(logz);
            a = zeros(self.dim,T);
            
            a(:,1) = like(:,1).*pi;            
%            [a(:,1),z(1)] = util.normalize(a(:,1),z(1));
            a_sum = sum(a(:,1),1);
            a(:,1)=a(:,1)/a_sum;
            
            
%            z(1) = z(1)*a_sum;
            logz(1)=logz(1)+log(a_sum);
            
            for t = 2:T
                a(:,t) = like(:,t) .* (A'*a(:,t-1));
%                [a(:,t),z(t)] = util.normalize(a(:,t),z(t));
                a_sum = sum(a(:,t),1);
                a(:,t)=a(:,t)/a_sum;
%                z(t) = z(t)*a_sum;
                logz(t)=logz(t)+log(a_sum);
            end
                        
            % backward propogation
            b = zeros(self.dim,T);
            b(:,T) = 1/self.dim;
            for t = T:-1:2
                b(:,t-1) = A * (like(:,t) .* b(:,t));
                b(:,t-1) = b(:,t-1) / sum(b(:,t-1));
            end
            
            p = a .* b;
            p = bsxfun(@rdivide,p,sum(p,1));
            
            if(issparse(A))
               [i,j,Atemp]=find(A); 
               xi=a(i,1:end-1).*b(j,2:end).*like(j,2:end).*Atemp;
               xi=xi./sum(xi,1);
               xi(isnan(xi(:)))=0;
               SExx = sum(xi,2);
               SExx = sparse(i,j,SExx);
            else
                xi = bsxfun(@times,permute(a(:,1:end-1),[1,3,2]), ...
                    permute(b(:,2:end).*like(:,2:end),[3,1,2]));            
                xi = bsxfun(@times,xi,A);
                xi = bsxfun(@rdivide,xi,sum(sum(xi,1),2));
                xi(isnan(xi(:)))=0;
                SExx = squeeze(sum(xi,3));
            end
%            logZ = sum(log(z));
            logZ = sum(logz);
        end
                
        function res = KLqprior(self)
            res = + self.A.KLqprior + self.pi0.KLqprior;
            for i=1:self.dim
                res = res + self.obsModels{i}.KLqprior;
            end
        end
        
        function initialize(self,data)
            datacat=[data{:}]';
            z=kmeans(datacat,self.dim);
            for i=1:self.dim
                self.obsModels{i}.update(datacat(z==i,:),1);
                for j=1:self.dim
                    xi(i,j,1) = sum(z(1:end-1)==i & z(2:end)==j);
                end
            end
            self.A.update(xi/length(data));

        end
        
        function res = state_means(self)
            for k=1:self.dim
                res(k,:) = self.obsModels{k}.mean;
            end
        end
        
        function plot(self,data,fignum,nf)
            figure(fignum)
            clf
            cc=jet(self.dim);
            if(~exist('fignum','var')) fignum=1; end
            imagesc(self.A.mean-diag(diag(self.A.mean)));
            for j=1:min(numel(data),nf)
                [m,idx]=max(self.p{j});

                len=size(data{j},2);
                
                figure(fignum+j)
                scatter(data{j}(1,:),data{j}(2,:),3*ones(1,len),cc(idx,:))
                drawnow

            end
        end 
        
        function [res,Y]=VarExplained(self,data)
            % Assumes that p is up to date
            idx=[];
            for i=1:length(self.obsTypes)
                idx=[idx,self.obsTypes{i}.idx];
            end
            for i=1:self.dim
                mu(:,i)=self.obsModels{i}.mean;
            end
            mu=mu(idx,:);
            for i=1:length(data)
                Y{i}=mu*self.p{i};
                res(i,:)=1-mean((Y{i}'-data{i}(idx,:)').^2)./(var(data{i}(idx,:)'));
            end
        end

% DO NOT USE IT IS FUCKING SLOWER THAN SHIT.
%         function [p,SExx,logZ]=logforwardbackward(self,loglike,logA,logpi)
%             T=size(loglike,2);
%             loga = zeros(self.dim,T);
%             logb = zeros(self.dim,T);
%             logZ = zeros(1,T);
%             
%             loga(:,1) = loglike(:,1) + logpi;
%             [loga(:,1),logZ(1)] = util.lognormalize(loga(:,1),1);
%             logb(:,T) = util.lognormalize(logb(:,T),1);
%             
%             for t = 2:T
%                 loga(:,t) = loglike(:,t) + ...
%                     util.logsumexp(bsxfun(@plus,loga(:,t-1),logA)');
%                 [loga(:,t),logZ(t)] = util.lognormalize(loga(:,t),1);
%                 
%                 logb(:,T-t+1) = util.logsumexp(bsxfun(@plus, ...
%                     logA,loglike(:,T-t+2)' + logb(:,T-t+2)'));
%                 logb(:,T-t+1) = util.lognormalize(logb(:,T-t+1),1);
%             end
%             
% %            p = exp(loga + logb);
% %            p = bsxfun(@rdivide,p,sum(p,1));
%             p = exp(util.lognormalize(loga + logb,1));
%             
%             logxi = bsxfun(@plus,permute(loga(:,1:end-1),[1,3,2]), ...
%                 permute(logb(:,2:end) + loglike(:,2:end),[3,1,2]));
%             logxi = bsxfun(@plus,logxi,logA);
%             xi = exp(logxi);
%             xi = bsxfun(@rdivide,xi,sum(sum(xi,1),2));
%             xi(isnan(xi(:))) = 0;
%             SExx = squeeze(sum(xi,3));
%             logZ = sum(logZ);
%         end

    end
end