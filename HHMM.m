classdef HHMM < handle
    properties
        % Two level heirarchal hidden markov model.  Ultimately this is
        % simply a wrapper for the HMM that encodes transitions
        % probabilities in a heirarchia form.  High level states are given
        % by Qp and associated low level state are given by p.  The bulk of
        % the inference is done in the HMMequiv object.

        % As with HMM observations can take the form of a generalized
        % product of exponential family mambers specified usinng obsTypes.
        
        % See GPEF.m for more information
        
        Qdim % number of high level states
        Q % transition probabilities at highest level
        Qpi0 % initial high level state distribution

        dim % size of low level state space
        A   % A{i} is the transition probability matrix between low level 
            % states when in high level state i
        pi0 % pi0{i} is the initial state distribution after a highlevel transition
            % to state i occurs. 
        Aidx % relates Q to HMMequiv.  The idea is to give HMMequiv.A block 
             % structure that groups Q states, i.e. diag(A{k}*Q(k,k)) 
             
        D % dimension of the observation space
        
        obsTypes % obsTypes is input to the constructor of the dists.GPEF distribution
                 %
                 % obsTypes{k}.dist = 'mvn','normal','poisson','gamma','mult','binary'
                 % types{k}.idx indicate the dimensions of the data matrix associated 
                 % with that data type.  For the purposes of handling missing data
                 % two independent poisson variabels should have a different
                 % entries in the obsTypes cell array.  
        obsModels 
        
        HMMequiv
        p
        Qp
        Exx
        logZ
        L
    end

    methods
        function self = HHMM(Qdim,dim,D,obsTypes,Qalpha_0, Qpi0alpha_0,Aalpha_0,pi0alpha_0)
            self.Qdim = Qdim;
            self.dim = dim;
            self.D = D;
            self.Q=dists.transition(dim,Qalpha_0);
            self.Qpi0=dists.expfam.dirichlet(dim,Qpi0alpha_0);

            for i = 1:Qdim
                self.Aidx{i}=[1:dim]+(i-1)*dim;
                self.A{i}  = dists.transition(dim,Aalpha_0);
                self.pi0{i}= dists.expfam.dirichlet(dim,pi0alpha_0);
            end
            
            if(isempty(obsTypes))
                fprintf('Defaulting to normally distributed observations.\n');
                self.obsTypes{1}.dist = 'mvn';
                self.obsTypes{1}.idx = [1:D];
            elseif(ischar(obsTypes))                
            else
                self.obsTypes = obsTypes;
                for i=1:dim
                for j=1:Qdim
                    self.obsModels{i,j}=dists.GPEF(self.obsTypes);
                end
                end
            end            
            
            for i=1:Qdim
            for j=1:Qdim
                Aalpha_0temp(self.Aidx{i},self.Aidx{j}) = Qalpha_0(i,j)*Aalpha_0;
            end
                pi0alpha_0temp(self.Aidx{i}) = Qpi0alpha_0(i)*pi0alpha_0;
            end
            
            self.HMMequiv=HMM(self.Qdim*self.dim,D,obsTypes,Aalpha_0temp,pi0alpha_0temp);
            self.HMMequiv.obsModels=self.obsModels(:);
            
        end
        
        function update_states(self,data,obsloglike,didx)
            if(~exist('obsloglike','var')) 
                obsloglike = self.HMMequiv.obsloglike(data);
            end
            if(~exist('didx','var'))
                didx=[1:numel(data)];
            end
                        
            logQ = self.Q.loggeomean;
            logQpi0 = self.Qpi0.loggeomean;
            logpi0 = zeros(self.Qdim*self.dim,1);
            for k=1:self.Qdim
                logA(self.Aidx{k},self.Aidx{k}) ...
                    = logQ(k,k)+self.A{k}.loggeomean;
                logpi0(self.Aidx{k},1) = logQpi0(k,1) + self.pi0{k}.loggeomean;
                
                for l=[1:k-1,k+1:self.Qdim]
                    logA(self.Aidx{k},self.Aidx{l}) ...
                        = logQ(k,l)+repmat(self.pi0{l}.loggeomean',self.dim,1);
                end
            end

            for i=didx % can be parallelized 
                [p,Exx,logZ] = self.HMMequiv.forwardbackward(obsloglike{i},logA,logpi0);
                self.p{i} = p;
                self.Exx{i} = Exx;
                self.logZ{i} = logZ;                             
                self.Qp{i}=zeros(self.Qdim,size(p,2));
                for j=1:self.Qdim
                    self.Qp{i}(j,:)=sum(p(self.Aidx{j},:),1);
                end
            end
            self.p = self.p(1:length(data));
            self.Qp = self.Qp(1:length(data));
            self.Exx = self.Exx(1:length(data));
            self.logZ = self.logZ(1:length(data));
            
        end

        function L = update(self,data)
            self.update_states(data);  
            self.L = sum(cellfun(@sum,self.logZ)) - self.KLqprior;
                %SExx and SEx0 are stored in HMMequiv format                
            L=self.L;
            
            SExx = zeros(self.HMMequiv.dim,self.HMMequiv.dim);
            SEx0 = zeros(self.HMMequiv.dim,1);
            for i=1:numel(data);
                SExx = SExx + self.Exx{i};
                SEx0 = SEx0 + self.p{i}(:,1);
            end
            
            for k=1:self.Qdim
                temp0(k)=sum(SEx0(self.Aidx{k}));
            for l=1:self.Qdim
                temp(k,l) = sum(sum(SExx(self.Aidx{k},self.Aidx{l})));
            end
            end
            self.Q.update(temp);
            self.Qpi0.update(temp0');
            
            for k=1:self.Qdim
                self.A{k}.update(SExx(self.Aidx{k},self.Aidx{k}));
                temp = zeros(self.dim,self.dim);
                for l=[1:k-1,k+1:self.Qdim]
                    temp = temp + SExx(self.Aidx{l},self.Aidx{k});
                end
                temp=sum(temp,1);
                self.pi0{k}.update(SEx0(self.Aidx{k})+temp');
            end
            
            datacat = [data{:}];
            pcat = [self.p{:}];
            for k=1:self.dim*self.Qdim
                if(sum(pcat(k,:))>1)
                    self.HMMequiv.obsModels{k}.update(datacat',pcat(k,:)');
                else
                    self.HMMequiv.obsModels{k}.update([],0);
                end
            end
        end
        
        function updateMarkovparms(self,data,p,didx)
           if(~exist('p','var'))
                p=ones(numel(data),1); 
           elseif(isempty(p))
                p=ones(numel(data),1);                
           else
                n=sum(p);
           end
            if(~exist('didx','var'))
                didx=[1:numel(data)];
            end
            SExx = zeros(self.HMMequiv.dim,self.HMMequiv.dim);
            SEx0 = zeros(self.HMMequiv.dim,1);
            pcat=cell(size(data));
            for i=didx;
                SExx = SExx + self.Exx{i}*p(i);
                SEx0 = SEx0 + self.p{i}(:,1)*p(i);
                pcat{i} = self.p{i}*p(i);
            end
            
            for k=1:self.Qdim
                temp0(k)=sum(SEx0(self.Aidx{k}));
            for l=1:self.Qdim
                temp(k,l) = sum(sum(SExx(self.Aidx{k},self.Aidx{l})));
            end
            end
            self.Q.update(temp);
            self.Qpi0.update(temp0');
            
            for k=1:self.Qdim
                self.A{k}.update(SExx(self.Aidx{k},self.Aidx{k}));
                temp = zeros(self.dim,self.dim);
                for l=[1:k-1,k+1:self.Qdim]
                    temp = temp + SExx(self.Aidx{l},self.Aidx{k});
                end
                temp=sum(temp,1);
                self.pi0{k}.update(SEx0(self.Aidx{k})+temp');
            end
 
        end
        
        
        function updateparms(self,data,p) 
            % assumes states are up to date.  
            if(~exist('p','var'))
                p=ones(numel(data),1); 
            else
                n=sum(p);
            end
            SExx = zeros(self.HMMequiv.dim,self.HMMequiv.dim);
            SEx0 = zeros(self.HMMequiv.dim,1);
            pcat=cell(size(data));
            for i=1:numel(data);
                SExx = SExx + self.Exx{i}*p(i);
                SEx0 = SEx0 + self.p{i}(:,1)*p(i);
                pcat{i} = self.p{i}*p(i);
            end
            
            for k=1:self.Qdim
                temp0(k)=sum(SEx0(self.Aidx{k}));
            for l=1:self.Qdim
                temp(k,l) = sum(sum(SExx(self.Aidx{k},self.Aidx{l})));
            end
            end
            self.Q.update(temp);
            self.Qpi0.update(temp0');
            
            for k=1:self.Qdim
                self.A{k}.update(SExx(self.Aidx{k},self.Aidx{k}));
                temp = zeros(self.dim,self.dim);
                for l=[1:k-1,k+1:self.Qdim]
                    temp = temp + SExx(self.Aidx{l},self.Aidx{k});
                end
                temp=sum(temp,1);
                self.pi0{k}.update(SEx0(self.Aidx{k})+temp');
            end
            
            datacat = [data{:}];
            if(~isempty(pcat))
                pcat = [pcat{:}];
            else
                pcat=zeros(self.HMMequiv.dim,1);
            end
            for k=1:self.HMMequiv.dim
                self.HMMequiv.obsModels{k}.update(datacat',pcat(k,:)');
            end
        end
                
        function res = Eloglikelihood(self,data)
            if(~exist('data','var'))
                res = self.logZ;
            else
                self.update_states(data);
                res = self.logZ;
            end
        end
             
        function res = KLqprior(self)
            res = self.Q.KLqprior + self.Qpi0.KLqprior;
            for i=1:self.dim
            for j=1:self.Qdim
                res = res + self.obsModels{i,j}.KLqprior;
            end
            end
            for i=1:self.Qdim
                res = res + self.A{i}.KLqprior;
                res = res + self.pi0{i}.KLqprior;
            end
        end
        
        function initialize(self,data)
            datacat=[data{:}]';
            z=kmeans(datacat,self.Qdim);
    
            for i=1:self.Qdim
                idx=find(z==i);
                z2=kmeans(datacat(idx,:),self.dim);
                for j=1:self.dim
                    idx2=find(z2==j);
                    self.obsModels{j,i}.update(datacat(idx(idx2),:));
                end
            end
        end
        
        function plot(self,data,fignum,nf)
            cc=jet(self.Qdim);
            if(~exist('fignum','var')) fignum=1; end
            figure(fignum)
            clf
            imagesc(self.Q.mean-diag(diag(self.Q.mean)))
            for j=1:min(numel(data),nf)
                [m,idx]=max(self.Qp{j});

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
            for i=1:self.dim*self.Qdim
                mu(:,i)=self.obsModels{i}.mean;
            end
            mu=mu(idx,:);
            for i=1:length(data)
                Y{i}=mu*self.p{i};
                res(i,:)=1-mean((Y{i}'-data{i}(idx,:)').^2)./(var(data{i}(idx,:)'));
            end
        end

    end
end