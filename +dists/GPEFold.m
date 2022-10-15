classdef GPEF < handle  %COMMENT this shit would be so much easier with pointers
    properties
        % hyperparameters
        D    % total dimension of data matrix
        dims % dims(k) indicates the number of variables of type k.
        type % type{k} is a string indicating the type of random variable 
             % in dimension k of the associated data matrix.  All are
             % assumed independent except for MVN and multinomial  
             % Legit options are:
             % mvn, normal, poisson, gamma, multinomial.
             % Note that the actual distributions used are the associated
             % conjugate priors:  NW, normalinvgamma, gamma, gammagamma,
             % dirichlet

        idx  % idx{n}, n=1..5 indicates which dimensions correspond to each 
             % data type.  n=1 -> mvn, n=2 -> normal, n=3 -> poisson, 
             %             n=4 _> gamma, n=5 -> mult.     
        NW
        normalinvgamma
        poissgamma
        gammagamma
        dirichlet
        beta
    end
    
    methods 
        
        function self = GPEF(labels)
            
            self.dims = length(labels);
            self.idx = cell(1,6);
            for k=1:self.dims
                switch labels{k}
                    case 'mvn'
                        self.idx{1} = [self.idx{1},k];
                    case 'normal'
                        self.idx{2} = [self.idx{2},k];
                    case 'poisson'
                        self.idx{3} = [self.idx{3},k];
                    case 'gamma'
                        self.idx{4} = [self.idx{4},k];
                    case 'multinomial'
                        self.idx{5} = [self.idx{5},k];
                    case 'binary'
                        self.idx{6} = [self.idx{6},k];
                end
            end
            if(sum(cellfun(@length,self.idx))<self.dims)
               'warning: some data dimensions not assigned valid labels' 
            end
                
            D = length(self.idx{1});
            self.dims(1)=D;
            self.NW = dists.expfam.NW(zeros(D,1),1,eye(D),1);  
                        
            D=length(self.idx{2});
            self.dims(2)=D;
            self.normalinvgamma=dists.expfam.normalinvgamma(zeros(D,1),ones(D,1),ones(D,1),ones(D,1));
            
            D = length(self.idx{3});
            self.dims(3)=D;
            self.poissgamma = dists.poissgamma(ones(D,1),ones(D,1));
            
            D = length(self.idx{4});
            self.dims(4)=D;
            self.gammagamma = dists.gammagamma(ones(D,1),ones(D,1),ones(D,1),ones(D,1));
            
            D = length(self.idx{5});
            self.dims(5)=D;
            self.dirichlet = dists.expfam.dirichlet(D,ones(D,1));
            
            D = length(self.idx{6});
            self.dims(6)=D;
            self.beta = dists.expfam.betadist(ones(D,1),ones(D,1));


        end
        
        function update(self,data,p)  
            
            % data is the raw data matrix of size ns by 
            % if present p is an assignment probability (for clustering)
            % and is assumed to be a vector

            % NW
            if(~exist('p','var'))
               p=ones(size(data,1),1); 
            end
            n=sum(p);
            
            if(self.dims(1)>0)
                Ex = p'*data(:,self.idx{1})/n;
                Exx = data(:,self.idx{1})'*bsxfun(@times,data(:,self.idx{1}),p)/n;
                self.NW.update(Ex',Exx,n);
            end
            
            % Independent normal
            if(self.dims(2)>0)
                Ex = p'*data(:,self.idx{2})/n;
                Exx = p'*(data(:,self.idx{2}).^2)/n;    
                self.normalinvgamma.update(Ex',Exx',n)
            end
            
            % gamma-poisson
            if(self.dims(3)>0)
                Ex = p'*data(:,self.idx{3});
                self.poissgamma.update(Ex'/2,n/2);
            end
            
            % Gamma
            if(self.dims(4)>0)
                Ex = p'*data(:,self.idx{4})/n;
                Elogx = p'*log(data(:,self.idx{4}))/n;
                self.gammagamma.update(Ex',Elogx',n);
            end
            
            % dir-mult
            if(self.dims(5)>0)
                Ex = p'*data(:,self.idx{5});
                self.dirichlet.update(Ex');
            end
            
            % beta-binary
            if(self.dims(6)>0)
                Ex = p'*data(:,self.idx{6});
                self.beta.update(Ex',n-Ex');
            end
        end                                
        
        function res = entropy(self)
            res=0;
            if(self.dims(1)>0) res = res + self.NW.entropy; end
            if(self.dims(2)>0) res = res + self.normalinvgamma.entropy; end
            if(self.dims(3)>0) res = res + self.poissgamma.entropy; end
            if(self.dims(4)>0) res = res + self.gammagamma.entropy; end
            if(self.dims(5)>0) res = res + self.dirichlet.entropy; end                
            if(self.dims(6)>0) res = res + self.beta.entropy; end                
        end
        
        function res = KLqprior(self)
            res=0;
            if(self.dims(1)>0) res = res + self.NW.KLqprior; end
            if(self.dims(2)>0) res = res + self.normalinvgamma.KLqprior; end
            if(self.dims(3)>0) res = res + self.poissgamma.KLqprior; end
            if(self.dims(4)>0) res = res + self.gammagamma.KLqprior; end
            if(self.dims(5)>0) res = res + self.dirichlet.KLqprior; end                
            if(self.dims(6)>0) res = res + self.beta.KLqprior; end                
        end
        
        function res = Eloglikelihood(self,data)
            res=0;
            if(self.dims(1)>0) res = res + self.NW.Eloglikelihood(data(:,self.idx{1})); end
            if(self.dims(2)>0) res = res + self.normalinvgamma.Eloglikelihood(data(:,self.idx{2})); end
            if(self.dims(3)>0) res = res + self.poissgamma.Eloglikelihood(data(:,self.idx{3})); end
            if(self.dims(4)>0) res = res + self.gammagamma.Eloglikelihood(data(:,self.idx{4})); end
            if(self.dims(5)>0) res = res + self.dirichlet.Eloglikelihood(data(:,self.idx{5})); end                            
            if(self.dims(6)>0) res = res + self.beta.Eloglikelihood(data(:,self.idx{6})); end                            
        end

    end
end
