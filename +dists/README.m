% dists Directory contains a set of derived distributions that can serve as guides 
% for VB inference.  
%
% All distributions can be updated using computed expecttations of their 
% associated sufficient statistics using "dist".updateSS routines.
%
% "dist".KLqprior gives the KL divergence between prior and posterior which 
% is used to compute the distributions contribution to the ELBO
% 
% "dist".E"" commputes relevant expectations of the sufficneit statistics
% and a few others.
% 
% Some routines innclude "dist".rawupdate(data,p) which takes in raw data 
% and a p which is the probability that a given data point belongs to the
% distribution in questions.  This is done to make mixture modeling easier
%
% Inside the rawupdate routine expectations of sufficient statistics are computed using
%     E<T(x)>_p = sum(T(x)*p)
%
% Everything has been vectorized, but not consistently.  Some distributions
% have specialized versions designed for specific cases where algorithms
% might prefer that a dirichlet be given in a row instead of a column.  So
% its always worth checking that the mean routinn has the desired format. 
%
% The expfam directory contains exponential family distributions which all 
% have exact update routines.  I.e. perfect coordinate ascent updates.  Not all
% derived routines have this property.  For example, normalsparse uses the
% automatic relevance determination trick so to update its parameters
% requires a few VB iterations.  When usinng derived routines of this form, the ELBO 
% is not guarenteed to monotinically converge, but experience shows that deviations 
% are typically small and usually only occur at the very beginning and end of learning. 