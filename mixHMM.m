classdef mixHMM < handle

    % Implements a mixture of heirarchial hidden markov models.  
    % p(i,j) encodes the probability that time series i is modeled by HMM j
    % For the Fly movement data sets, this can be interpreted as the
    % probability that fly i is using strategy j.  

    properties
        NC  % number of clusters (each cluster is defined by a HMM)
        dim % dimension of the state space
        D % dimension of the observation space
        
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  
        HMMs
        
        pi
        
        p
        NA
        logptilde
        L
    end

    methods
        
        function self = mixHMM(NC,dim,D, obsTypes, alpha_0, Aalpha_0, pi0alpha_0)
            if(isempty(obsTypes))
                obsTypes{1}.dist='mvn';
                obsTypes{1}.idx=[1:D];
                self.D = D;
            else
                self.obsTypes = obsTypes;
                self.D = 0;
                for i=1:length(obsTypes)
                    self.D = self.D + length(obsTypes{i}.idx);
                end                
            end
            
            self.NC = NC;
            self.dim = dim;
            
            for i=1:NC
                self.HMMs{i}=HMM(dim,D,obsTypes,Aalpha_0,pi0alpha_0);
            end
            
            self.pi=dists.expfam.dirichlet(NC,alpha_0);
            
            
        end
                
        function updateassignments(self,data)
            self.logptilde=zeros(self.NC,length(data));
            for i=1:self.NC
                self.HMMs{i}.Eloglikelihood(data);  % updates states;
                self.logptilde(i,:) =  cell2mat(self.HMMs{i}.logZ); 
            end
            % add prior
            self.logptilde = bsxfun(@plus,self.logptilde,self.pi.loggeomean);
            
            % normalize
            self.p = exp(bsxfun(@minus,self.logptilde,max(self.logptilde)));            
            self.p = bsxfun(@rdivide,self.p, sum(self.p,1));
            
        end
        
        function L = update(self,data,modeliters)            
            if(~exist('modeliters','var'))
                modeliters=1;
            end
% Update Assignments
            
            for i=1:self.NC
                self.HMMs{i}.update_states(data);
                self.logptilde(i,:) =  cell2mat(self.HMMs{i}.logZ); 
            end
            % add prior
            self.logptilde = bsxfun(@plus,self.logptilde,self.pi.loggeomean);
            
            % normalize
            self.p = exp(bsxfun(@minus,self.logptilde,max(self.logptilde)));            
            self.p = bsxfun(@rdivide,self.p, sum(self.p,1));
            
% Compute Lower bound
            self.L = -self.KLqprior;
            idx=find(self.logptilde(:)>-Inf);
            self.L = self.L + sum(self.p(idx).*self.logptilde(idx));
            idx = find(self.p(:)>0);
            self.L = self.L - sum(self.p(idx).*log(self.p(idx)));
            L = self.L;
            self.NA = sum(self.p,2);
            self.pi.update(self.NA);
            
            for i=1:self.NC
                if(self.NA(i)>1)
                    self.HMMs{i}.updateparms(data,self.p(i,:));
                else
                    self.HMMs{i}.updateparms({},0);
                end
            end
            
            for j=2:modeliters
                for i=1:self.NC
                    if(self.NA(i)>1)
                        self.HMMs{i}.update_states(data);
                        self.HMMs{i}.updateparms(data,self.p(i,:));
                    else
                        self.HMMs{i}.updateparms({},0);
                    end
                end                
            end
            
        end
        
        function merge(self,data,iters,i,j) %only works after a call to update
            
            idx=find(self.NA>1);
            if(length(idx)<2) %do nothing
                fprintf('no possible merges\n')
            else                
                if(~exist('j','var'))
                    idx=idx(randperm(length(idx)));
                    i=idx(1);
                    j=idx(2);
                end
                psave = self.p;
                NAsave = self.NA;
                Lsave = self.L;
                alphasave = self.pi.alpha;
                self.p(:,i) = (self.p(:,j)+self.p(:,i));
                self.p(:,j) = 0;
                self.NA(i)=self.NA(i)+self.NA(j);
                self.NA(j)=0;

                HMMi = self.HMMs(i);
                HMMj = self.HMMs(j);
                for j=1:iters
                    self.HMMs{i}.update_states(data);
                    self.HMMs{i}.updateparms(data,self.p(i,:));
                end
                self.HMMs{j}=HMM(self.dim,self.D,self.obsTypes,self.HMMs{j}.Aalpha_0,self.HMMs{j}.pi0alpha_0);
                
                self.update(data);
                if(self.L <= Lsave) % reject merge
                    'merge rejected'
                    self.p = psave;
                    self.NA = NAsave;
                    self.L = Lsave;                    
                    self.pi.alpha = alphasave;
                    self.HMMs(i)=HMMi;
                    self.HMMs(j)=HMMj;
                end
            end

            
        end
        
        function res = KLqprior(self)
            res = self.pi.KLqprior;
            for i=1:self.NC
                res = res + self.HMMs{i}.KLqprior;
            end
        end
        
        function initialize(self,data,modeliters,z)
            for i=1:length(data)
                len(i)=size(data{i},2);
            end
            T=min(len);
            idx=[];
            for i=1:length(self.obsTypes)
                idx=[idx,self.obsTypes{i}.idx];
            end
            
            fprintf('Initializing...')
            for i=1:length(data)
                temp=data{i}(idx,end-T+1:end);               
                pruneddata(i,:) = temp(:);
            end
            if(~exist('z','var'))
                z = kmeans(pruneddata,self.NC);
            end
            
            self.p=repmat(1/self.NC^2,self.NC,length(data));
            for i=1:length(data) 
                self.p(z(i),i) = 1-1/self.NC+1/self.NC^2; 
            end

            for i=1:self.NC
                NA(i,1) = sum(z==i);
            end
            self.pi.update(NA);
            for j=1:modeliters
                for i=1:self.NC
%                     if(sum(self.p(i,:))>1)
%                         zobs = kmeans(pruneddata(z==i,:),self.dim);
%                         idx1=find(z==i);
%                         for k=1:self.dim
%                             idx2=find(zobs==k);
%                             self.HMMs{i}.obsModels{k}.update(pruneddata(idx1(idx2),:));
%                         end
%                         self.HMMs{i}.update(data(z==i));
%                     else
%                         self.HMMs{i}.updateparms({},0);
%                     end
                    self.HMMs{i}.update(data(z==i));
                end                
            end
            
            fprintf(['done.\n'])
            
        end
        
        function fillunused(self,data,modeliters,neff)            
            % can only be run after update
            if(~exist('neff','var'))
                neff=1;
            end
              
            idx=find(sum(self.p')<1);
            if(isempty(idx))
                return
            end
            fprintf(['Filling ',int2str(length(idx)),' unused clusters\n'])
            for i=1:self.NC
                fitq(i,:) = cell2mat(self.HMMs{i}.logZ);
            end
            m=max(fitq);
            [m,didx]=sort(-m);
            datatemp = data(didx(1:length(idx)));
            for i=1:length(idx)
                for j=1:modeliters
                    self.HMMs{idx(i)}.update_states(datatemp(i));
                    self.HMMs{idx(i)}.updateMarkovparms(datatemp(i),neff);
                end
            end
            self.pi.alpha=self.pi.alpha_0;
        end
        
        function split(self,data,modeliters,cidx)
            % can only be run after update
            % Choose a cluster to split based upon size.
            
            if(~exist('cidx','var'))
                idx1 = util.discretesample(  self.pi.mean,1);
            else
                idx1=cidx;
            end
           
            % find 2 smallest cluster to replace
            NA = sum(self.p');
            [m,idx2]=sort(NA);
            idx2=idx2(1);
            
            % Find datapoints assigned to that cluster
            
            [m,pidx]=max(self.p);
            pidx = find(pidx==idx1);
            
            if(length(pidx)<2)
                return
            end
            % Cluster empirical state distributions
            %
            temp=zeros(self.dim,length(pidx));
            for i=1:length(pidx)
                temp(:,i) = sum(self.HMMs{idx1}.p{pidx(i)},2);
            end
            z=kmeans(temp',2);
            
            datatemp1 = data(pidx(z==1));
            datatemp2 = data(pidx(z==2));
            if(isempty(datatemp1) | isempty(datatemp2))
                return
            end
            self.HMMs{idx1}.A.alpha = self.HMMs{idx1}.A.alpha_0;
            self.HMMs{idx1}.pi0.alpha = self.HMMs{idx1}.pi0.alpha_0;
            
            for j=1:modeliters
                self.HMMs{idx1}.update(datatemp1);
                self.HMMs{idx2}.update(datatemp2);
            end
            
%            self.pi.alpha(idx2)=self.pi.alpha(idx1)/2;
%            self.pi.alpha(idx1)=self.pi.alpha(idx1)/2;
            self.pi.alpha=self.pi.alpha_0;

        end
                
            
        function prune(self)
            idx=find(self.NA>0.5);
            if(length(idx)==self.NC)
                fprintf('no unused clusters')
                return
            end                
            self.NC = length(idx)+1;
            for i=1:length(idx)
                HMMs(i)=self.HMMs(idx(i));
            end
            idx=find(self.NA<=0.5);
            HMMs(i+1)=self.HMMs(idx(1));
            
            self.pi=dists.expfam.dirichlet(self.NC,self.pi.alpha_0(1:self.NC,1));
            self.HMMs=HMMs;
            self.logptilde = self.logptilde(1:self.NC,:);

            
        end
        
        function plotclusters(self,data,n,fignum)
            figure(fignum)
            clf
            cc=jet(self.dim);
            [m,idx]=max(self.logptilde);
            if(~exist('fignum','var')) fignum=1; end
            if(~exist('n','var')) n=20; end
            for j=1:self.NC
                idxj = find(idx==j);
                
                [mj,idxj2]=sort(-self.logptilde(j,idxj));
                idxj = idxj(idxj2);
                
                if(length(idxj)>n)
                    idxj = idxj(1:n);
                end

                d1=data(idxj);
                p1=self.HMMs{j}.p(idxj);
    
                px = ceil(sqrt(self.NC));
                py = ceil(self.NC/px);
                
                if(length(d1)>0)
                    subplot(px,py,j)
                    hold on
                    for i=1:length(d1)
                        [m,idx2]=max(p1{i});
                        scatter(d1{i}(2,:),d1{i}(3,:),3*ones(size(d1{i}(3,:))),cc(idx2,:))
                        scatter(2-d1{i}(2,:),d1{i}(1,:),3*ones(size(d1{i}(3,:))),cc(idx2,:))
                    end    
                    hold off
                end
            end
            drawnow
        end 
    end
end