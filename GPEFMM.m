classdef GPEFMM < handle

    properties
        % Parameters
        Nc % number of clusters
        D  % dimension of the observations
        L % lower bound
        alpha_0
        minclustersize
        
        % Observation Types
        ObsTypes
        % 
        
        
        p  % assignment probabilities (N x Nc)
        logptilde
        NA % number of assignments to each class
        iters

        pi 
        dist  % GPEF cluster distributions
    end
    
    methods
        function self = GPEFMM(Nc,D,alpha_0,ObsTypes)
            if(~exist('ObsTypes','var'))
                % assume mvn and idx=1:D
                fprintf('Types not specified.  Defaulting to MVN\n')
                self.ObsTypes{1}.dist='mvn';
                self.ObsTypes{1}.idx=[1:D];
            else
                self.ObsTypes=ObsTypes;
            end
            self.Nc = Nc;
            self.D = D;
            self.alpha_0 = alpha_0/Nc;
                
            self.minclustersize = 0.1;
                    
             self.L = -Inf;
             self.pi=dists.expfam.dirichlet(self.Nc,self.alpha_0*ones(1,self.Nc));
             for i=1:Nc
                 self.dist{i}=dists.GPEF(self.ObsTypes);
             end
             self.iters=0;
        end
        
        function fit(self,X,tol,maxiters,Ncguess)
            if(~exist('Ncguess','var'))
                Ncguess=self.Nc;
            elseif(Ncguess>self.Nc)
                for i=self.Nc+1:Ncguess
                    self.dist{i}=dists.GPEF(self.ObsTypes);
                end
                self.alpha_0 = self.alpha_0*self.Nc/Ncguess;
                self.Nc = Ncguess;
                self.pi=dists.expfam.dirichlet(self.Nc,self.alpha_0*ones(1,self.Nc));
            elseif(Ncguess<self.Nc/2)
                self.alpha_0 = self.alpha_0*self.Nc/(Ncguess+2);
                self.Nc = Ncguess+2;
                self.pi=dists.expfam.dirichlet(self.Nc,self.alpha_0*ones(1,self.Nc));
                self.dist=[];
                for i=1:self.Nc
                    self.dist{i}=dists.GPEF(self.ObsTypes);             
                end
            end
                        
            tic
            k=4;
            DL=self.update(X,4);
            Llast=-Inf;
            self.smartinitialization(X,Ncguess);
            self.update(X,2);                 
            while(k < maxiters && -DL/self.L > tol)
                k=k+4;
                DL = self.update(X,4);                 
                self.plotclusters(X,1)
            end
            if (k>=maxiters)
                fprintf('maximum iterations reached\n')
            else
                fprintf(['Discovered ',num2str(sum(self.NA>1)),' clusters after ',num2str(k),' iterations in ',num2str(toc),' seconds\n'])
            end
            fprintf(['Final <ELBO> = ',num2str(self.L),'\n'])
        end
        
        function smartinitialization(self,X,Ncguess)
            
            mu=mean(X);
            Y=zscore(X);
            
            [V,D]=eig(cov(Y),'vector');
            idx=cumsum(D)/sum(D)>0.05;
            Y=Y*V(:,idx)*diag(1./sqrt(D(idx)));
            
            [idx, muk] = kmeans(X, self.Nc);
            self.p=(idx==unique(idx)');
%            self.p=(self.p+1)/(1+self.Nc);
            self.NA = ones(1,self.Nc);
            self.logptilde = zeros(size(X,1),self.Nc);
            self.updateparms(X,0);
            
        end
        
        function DL = update(self,X,iters,fighandle)
            if(~exist('iters','var'))
                iters=1;
            end
            if(~exist('fighandle','var'))
                fighandle=0;
            end
            for i=2:iters
                self.updateparms(X,fighandle);
                self.updateassignments(X); 
                self.iters=self.iters+1;
            end
            self.updateparms(X,fighandle);
            DL = self.updateassignments(X); 
        end
        
        function DL = updateassignments(self,X)

            [N,D]=size(X);
            self.logptilde=zeros(N,self.Nc);
            for i=1:self.Nc
                self.logptilde(:,i) = self.dist{i}.Eloglikelihood(X);
            end
            self.logptilde = bsxfun(@plus,self.logptilde,self.pi.loggeomean);

            self.p = bsxfun(@minus,self.logptilde,max(self.logptilde')');            
            self.p = exp(self.p);
            self.p = bsxfun(@rdivide,self.p, sum(self.p,2));

            self.NA = sum(self.p,1);
            DL=self.L;
            self.L = - self.KLqprior;                                    
            self.L = self.L + sum(sum(self.p.*(self.logptilde)));
            idx = find(self.p(:)>0);
            self.L = self.L - sum(self.p(idx).*log(self.p(idx)));

            DL=self.L-DL;

        end
        
        function KL = KLqprior(self)            

            KL = self.pi.KLqprior;            
            for i=1:self.Nc
                KL = KL + self.dist{i}.KLqprior;
            end
                        
        end
        
        function updateparms(self,X,fighandle)
            if(isempty(self.p))
                self.updateassignments(X);
            end
            self.pi.update(self.NA);
            
            for i=1:self.Nc
                self.dist{i}.update(X,self.p(:,i));
            end
            
%             if(fighandle>0)
%                 self.plotclusters(X,fighandle)
%             end
        end
        
        function merge(self,X)
            idx=find(self.NA>1);
            if(length(idx)<2) %do nothing
                fprintf('no possible merges\n')
            else                
                idx=idx(randperm(length(idx)));
                i=idx(1);
                j=idx(2);
            
                psave = self.p;
                NAsave = self.NA;
                Lsave = self.L;
                self.p(:,i) = (self.p(:,j)+self.p(:,i));
                self.p(:,j) = 0;
                self.NA(i)=self.NA(i)+self.NA(j);
                self.NA(j)=0;

                self.updateparms(X,0);
                self.updateassignments(X);

                if(self.L <= Lsave) % reject merge
                    self.p = psave;
                    self.NA = NAsave;
                    self.updateparms(X,0);
                    self.L = Lsave;
                end
            end
        end
        
        function plotclusters(self,X,fighandle)
            figure(fighandle) 
            if(self.D<2)
                return
            end
%            idx0=find(self.NA>=1);
%            cc = rand(self.Nc,3);%jet(self.Nc);
                        
            [temp,idx] = max(self.p');
            clusters=unique(idx);
            cc0=jet(length(clusters));
            for i=1:length(clusters)
                cc(clusters(i),:)=cc0(i,:);
            end

            scatter(X(:,1),X(:,2),20*ones(size(X,1),1),cc(idx,:))
%             hold on
%             t=[0:1:100]/100*2*pi;
%             for i=1:self.Nc
%                 if(self.NA(i)> -1)
%                     C =(self.NWs{i}.ESigma);
%                     C=C(1:2,1:2);
%                     C=sqrtm(C);
%                 
%                     stdring = repmat(self.NWs{i}.mu(1:2),1,length(t)) + 2*C*[sin(t);cos(t)];
%                     plot(stdring(1,:),stdring(2,:),'color',cc(i,:))
% 
%                 end
%             end    
%             hold off
            title(strcat('<ELBO> = ',num2str(self.L/size(X,1))))

        end        
    
        function order_clusters(self)
            [m,loc]=sort(self.NA,'descend');
            self.dist = self.dist(loc);
            self.NA=self.NA(loc);
            self.pi.alpha=self.pi.alpha(loc);
        end
        
        function prune(self)
            self.order_clusters();
            Nc_new = sum(self.NA>1/2);
            self.Nc=Nc_new;
            self.dist = self.dist(1:Nc_new);
            self.p=[];
            self.logptilde=[];
            self.pi.alpha=self.pi.alpha(1:Nc_new);
            self.pi.alpha_0=self.pi.alpha_0(1:Nc_new);
        end
        
        function out = get_cluster_means(self)
            for i=1:self.Nc
                out(i,:) = self.dist{i}.mean;
            end
        end
    end
end

