classdef mixHHMM < handle
    
    % Implements a mixture of heirarchial hidden markov models.  
    % p(i,j) encodes the probability that time series i is modeled by HMM j
    % For the Fly movement data sets, this can be interpreted as the
    % probability that fly i is using strategy j.  
    
    properties
        NC  % number of clusters (each cluster is defined by a HMM)
        Qdim % dimension of the high level state space
        dim % dimension of the state space low level state space
        D % dimension of the observation space
        
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  

        HHMMs
        pi
        
        p
        NA
        logptilde
        L
    end

    methods
        
        function self = mixHHMM(NC,Qdim,dim,D, obsTypes, alpha_0, Qalpha_0, Qpi0alpha_0, Aalpha_0, pi0alpha_0)
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
            self.Qdim = Qdim;
            self.dim = dim;
            
            for i=1:NC
                self.HHMMs{i}=HHMM(Qdim,dim,D,obsTypes,Qalpha_0,Qpi0alpha_0,Aalpha_0,pi0alpha_0);
            end
            %self.pi=dists.expfam.dirichlet(NC,alpha_0/NC*ones(NC,1));
            self.pi=dists.expfam.dirichlet(NC,alpha_0/NC);
            
        end
                
        function updateassignments(self,data)
            self.logptilde=zeros(self.NC,length(data));
            for i=1:self.NC
                self.HHMMs{i}.Eloglikelihood(data);  % updates states;
                self.logptilde(i,:) =  cell2mat(self.HHMMs{i}.logZ); 
            end
            % add prior
            self.logptilde = bsxfun(@plus,self.logptilde,self.pi.loggeomean);
            idx=find(isnan(self.logptilde(:)));
            self.logptilde(idx)=-Inf;
            
            % normalize
            self.p = exp(bsxfun(@minus,self.logptilde,max(self.logptilde,[],1)));            
            self.p = bsxfun(@rdivide,self.p, sum(self.p,1));
            if(sum(isnan(self.p(:)))>0)
                ['NANs!!!!!']
                stop
            end
        end
        
        function DL = update(self,data,iters,modeliters)            
            if(~exist('modeliters','var'))
                modeliters=1;
            end
            if(~exist('iters','var'))
                iters=1;
            end
            
% Update Assignments
            
            for i=1:iters
                L=self.L;
                fprintf('updating assignments\n')
                self.updateassignments(data);
    % Compute Lower Bound
                self.L = -self.KLqprior;
                idx=find(self.logptilde(:)>-Inf);
                self.L = self.L + sum(self.p(idx).*self.logptilde(idx));
                idx = find(self.p(:)>0);
                self.L = self.L - sum(self.p(idx).*log(self.p(idx)));
                self.NA = sum(self.p,2);
                self.pi.update(self.NA);
    % Update Parms
                fprintf('updating parms\n')
                self.updateparms(data,modeliters)
            end
            DL=self.L-L;
        end
        
        function updateparms(self,data,modeliters)
            for i=1:self.NC
                self.HHMMs{i}.updateparms(data,self.p(i,:));
            end
            
            for j=2:modeliters
                for i=1:self.NC
                    self.HHMMs{i}.update_states(data);
                    self.HHMMs{i}.updateparms(data,self.p(i,:));
                end                
            end
            
        end
        
        function initialize(self,data,iters)
            idx=randperm(length(data));
            idx=idx(1:self.NC);                    
                
            self.p=ones(self.NC,length(data))/self.NC;
            
            for i=1:self.NC
                for j=1:iters
                    self.HHMMs{i}.update(data(idx(i)));
                end
            end
            self.logptilde=log(self.p);
            self.NA = sum(self.p,2);
            self.pi.update(self.NA);
        end
        
        
        function prune(self)
            idx=find(self.NA>0.25);
            NCold=self.NC;
            if(length(idx)>self.NC-2)
                fprintf('no empty clusters\n')
                return
            elseif(isempty(idx))
                fprintf('no data assigned to clusters\n')
                return
            else
                fprintf(['pruning ',num2str(self.NC-length(idx)-2),' clusters ',num2str(length(idx)+2),' clusters remain\n'])
            end            
            self.NC = length(idx)+2;
            for i=1:length(idx)
                HHMMs(i)=self.HHMMs(idx(i));
                NA(i,1)=self.NA(idx(i));
            end
            %above grabs non-empty clusters
            
            %below finds the two least empty clusters
            idx=find(self.NA<=0.25);
            [m,idx2]=sort(self.NA(idx),'descend');
            idx=idx(idx2(1:2));
            
            HHMMs(i+1)=self.HHMMs(idx(1));
            NA(i+1,1)=self.NA(idx(1));
            HHMMs(i+2)=self.HHMMs(idx(2));
            NA(i+2,1)=self.NA(idx(2));
            self.NA=NA;
            self.pi=dists.expfam.dirichlet(self.NC,self.pi.alpha_0(1:self.NC,1)*NCold/self.NC);
            self.pi.update(self.NA);
            
            self.HHMMs=HHMMs;
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
                HHMMj=self.HHMMs{j};
                self.HHMMs{j}=HHMM(self.Qdim,self.dim,self.D,self.obsTypes,self.HHMMs{j}.Q.alpha_0,self.HHMMs{j}.Qpi0.alpha_0,self.HHMMs{j}.A{1}.alpha_0,self.HHMMs{j}.pi0{1}.alpha_0);

                self.HHMMs{i}.updateparms(data,self.p(i,:));
                for k=2:iters
                    self.HHMMs{i}.update_states(data);
                    self.HHMMs{i}.updateparms(data,self.p(i,:));
                end
                                
                self.update(data);
                if(self.L <= Lsave) % reject merge
                    'merge rejected'
                    self.p = psave;
                    self.NA = NAsave;
                    self.L = Lsave;                    
                    self.pi.alpha = alphasave;
                    self.HHMMs{j}=HHMMj;                    
                end
                
                self.update(data)
            end
    
        end
        
        function res = KLqprior(self)
            res = self.pi.KLqprior;
            for i=1:self.NC
                res = res + self.HHMMs{i}.KLqprior;
            end
        end
                
        function fillunused(self,data,modeliters,NC2f)            
            % can only be run after update
            NCorig=self.NC;
            idx=find(self.NA<0.25);
            if(isempty(idx))
%                 fprintf('No Unused Clusters.  Doing Nothing\n')
%                 return
                fprintf('No Unused Clusters.  Adding one cluster to fill\n')
                alpha_0 = self.pi.alpha_0(1)*ones(self.NC+1,1)*self.NC;
                self.NC=self.NC+1;
                self.pi=dists.expfam.dirichlet(self.NC,alpha_0/self.NC);
                self.HHMMs{self.NC}=HHMM(self.Qdim,self.dim,self.D,self.obsTypes,self.HHMMs{1}.Q.alpha_0,self.HHMMs{1}.Qpi0.alpha_0,self.HHMMs{1}.A{1}.alpha_0,self.HHMMs{1}.pi0{1}.alpha_0);
                idx=self.NC;
            end
            
            if(~exist('NC2f','var'))
                NC2f=length(idx);
            elseif(NC2f>length(idx))
                NC2f=length(idx);
            end
            fprintf(['Filling ',int2str(NC2f),' unused clusters\n'])
            for i=1:NCorig
                fitq(i,:) = cell2mat(self.HHMMs{i}.logZ);
            end
            m=-max(fitq);
            %start new bit
            m=m-max(m);
            m=exp(m);
            m=m/sum(m);
            for i=1:NC2f
                didx(i)=util.discretesample(m,1);
                m=-max(fitq);
                m(didx(1:i))=-Inf;
                m=m-max(m);
                m=exp(m);
                m=m/sum(m);
            end
            
            for i=1:NC2f
                for j=1:modeliters
                    self.HHMMs{idx(i)}.update(data(didx(i)));
                end
            end
            self.pi.alpha=self.pi.alpha_0;
        end
        
        function split(self,data,modeliters,idx1)
            % can only be run after update
            % find smallest cluster to replace

            [m,idx2]=sort(self.NA);
            if(m(1)>1/2) 
               'split: no empty clusters'
               return
            end
            idx2=idx2(1);
            
            % Choose a cluster to split based upon size.
            psc=self.pi.mean;
            psc(idx2)=0;
            psc=psc/sum(psc);
                
            if(~exist('idx1','var'))
                idx1 = util.discretesample(  self.pi.mean,1);
            elseif(idx1==idx2)
                'cant split cluster with no elements'
                return
            end
            
            % Find datapoints assigned to that cluster
            
            [m,pidx]=max(self.p);
            pidx = find(pidx==idx1);
            
            if(length(pidx)<2)
                'cant split cluster with one element'
                return
            else
                ['splitting cluster ', num2str(idx1)]
            end
            % Cluster empirical state distributions
            %
            temp=zeros(self.dim*self.Qdim,length(pidx));
            for i=1:length(pidx)
                temp(:,i) = sum(self.HHMMs{idx1}.p{pidx(i)},2);
            end
            z=kmeans(temp',2);
            
            datatemp1 = data(pidx(z==1));
            datatemp2 = data(pidx(z==2));
            if(isempty(datatemp1) | isempty(datatemp2))
                return
            end
            for j=1:self.Qdim
                self.HHMMs{idx2}.A{j}.alpha = self.HHMMs{idx1}.A{j}.alpha;
                self.HHMMs{idx2}.pi0{j}.alpha = self.HHMMs{idx1}.pi0{j}.alpha;
            end
            self.HHMMs{idx2}.HMMequiv.pi0.alpha=self.HHMMs{idx1}.HMMequiv.pi0.alpha;
            self.HHMMs{idx2}.HMMequiv.A.alpha=self.HHMMs{idx1}.HMMequiv.A.alpha;
            self.HHMMs{idx2}.Q.alpha = self.HHMMs{idx1}.Q.alpha;
            self.HHMMs{idx2}.Qpi0.alpha = self.HHMMs{idx1}.Qpi0.alpha;
            
            
            for j=1:modeliters
                self.HHMMs{idx1}.update(datatemp1);
                self.HHMMs{idx2}.update(datatemp2);
            end
            
%            self.pi.alpha(idx2)=self.pi.alpha(idx1)/2;
%            self.pi.alpha(idx1)=self.pi.alpha(idx1)/2;
            self.pi.alpha=self.pi.alpha_0;

        end
                
        function plotclusters(self,data,fignum)
            figure(fignum)
            clf
            cc=jet(self.Qdim);
            [m,idx]=max(self.logptilde);
            if(~exist('fignum','var')) fignum=1; end
            for j=1:self.NC
                idxj = find(idx==j);
                
                [mj,idxj2]=sort(-self.logptilde(j,idxj));
                idxj = idxj(idxj2);
                if(numel(idxj)>0)
                    idxj=idxj(1);
                end
                clear p1 p2

                d1=data(idxj);
                p1=self.HHMMs{j}.p(idxj);
                
                for k=1:length(p1)
                for i=1:self.Qdim
                    p2{k}(i,:)=sum(p1{k}(self.HHMMs{j}.Aidx{i},:),1);
                end
                end
    
                px = ceil(sqrt(self.NC));
                py = ceil(self.NC/px);
                
                if(length(d1)>0)
                    subplot(px,py,j)
                    hold on
                    for i=1:length(d1)
                        [m,idx2]=max(p2{i});
                        scatter(d1{i}(1,:),d1{i}(2,:),3*ones(size(d1{i}(1,:))),cc(idx2,:))
                    end    
                    hold off
                end
            end
            drawnow
        end 
        
        function [res,Y]=VarExplained(self,data)
            [m,idx]=max(self.logptilde);
            for i=1:self.NC
                pve{i}=self.HHMMs{i}.VarExplained(data);
            end
            for i=1:length(data)
                res(i,:)=pve{idx(i)}(i,:);
            end
        end
        
    end
end