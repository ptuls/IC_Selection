function [x,r,bapp,IC,errHist] = threshold_fourier_spike(b,k,crit,errFcn,opts)
%THRESHOLD_FOURIER_SPIKE 
%   x = threshold_fourier_spike(b,k)
%   uses simple thresholding to select k Fourier and spike coefficients 
%   to estimate the solution to the equation
%
%       b = A*x     (or b = A*x + noise )
%
%   where there is prior information that x is sparse and A is the
%   dictionary of the Fourier and spikes. In Matlab, the full overcomplete
%   dictionary is equivalent to
%
%   A = [fft(eye(N))/sqrt(N) eye(N)], 
%
%   where N is the length of b. The first N atoms come from the Fourier  
%   basis, while the next N comes from the canonical Euclidean basis. Here,
%   note that x is twice the length of b, i.e. 2N.
%
%   [x,r,normR,bapp,resid,err] = threshold_fourier_spike(b,k,crit,errFcn,opts)
%   is the full version.
%
% Outputs:
%
%   'x' is the k-sparse estimate of the unknown signal
%   'r' is the residual b - A*x
%   'bapp' is the reconstructed signal approximation of b
%   'IC' minimum value of the information criterion selected (empty if 
%        no criterion was selected)
%   'resid' is a vector with final norm(r)
%   'err' is a vector with the output of errFcn
%
% Inputs:
%   
%   'b' is the vector of observations
%   'k' is the estimate of the sparsity (you may wish to purposefully
%       over- or under-estimate the sparsity, depending on noise)
%       N.B. k < size(A,1) is necessary, otherwise we cannot
%       solve the internal least-squares problem uniquely.
%
%   'k' (alternative usage):
%           instead of specifying the expected sparsity, you can specify
%           the expected residual. Set 'k' to the residual. The code
%           will automatically detect this if 'k' is not an integer;
%           if the residual happens to be an integer, so that confusion could
%           arise, then specify it within a cell, like {k}.
%
%   'crit' (optional) is the information criterion used: AIC, AICC, BIC, 
%           nMDL. If left empty, OMP will target k as the sparsity,
%           otherwise the final sparsity will be selected according to the
%           criterion used. Four are available:
%       
%           1. 'aic': Akaike information criterion
%           2. 'aicc': Akaike information criterion corrected for small
%                      samples
%           3. 'bic': Bayesian information criterion
%           4. 'mdl': two-stage Minimum Description Length
%           5. 'nmdl': normalized Minimum Description Length
%
%           Criterion is computed greedily, so choice may be suboptimal.
%
%   'errFcn'   (optional; set to [] to ignore) is a function handle
%              which will be used to calculate the error; the output
%              should be a scalar
%
%   'opts'  is a structure with more options, including:
%       .printEvery = is an integer which controls how often output is printed
%       .maxiter    = maximum number of iterations
%
%       Note that these field names are case sensitive!
%

% file:     threshold_fourier_spike.m, (c) Paul Tune, Jul 03 2015
% created: 	Fri Jul 03 2015 
% author:  	Paul Tune 
% email:   	paul.tune@adelaide.edu.au

% Modified version of OMP code by 
% Stephen Becker, Aug 1 2011.  srbecker@alumni.caltech.edu

%% Check options
n = length(b);  % length of signal

if nargin < 3
    crit = []; % no information criterion used
    IC = [];
else
    if ~isempty(crit)
        if strcmp(crit,'aic')
            fprintf('AIC criterion used\n');
        elseif strcmp(crit,'aicc')
            fprintf('AICC criterion used\n');
        elseif strcmp(crit,'bic')
            fprintf('BIC criterion used\n');
        elseif strcmp(crit,'mdl')
            fprintf('Two-stage MDL criterion used\n');
        elseif strcmp(crit,'nmdl')
            fprintf('Normalized MDL criterion used\n');
        else
           error('Not a valid information criterion');        
        end
        
        % maximum sparsity must be less than half the signal length
        k = floor(n/2);
        if ~mod(n,2)
            k = k-1;
        end  
    end
end

if nargin < 5, opts = []; end
if ~isempty(opts) && ~isstruct(opts)
    error('"opts" must be a structure');
end

function out = setOpts( field, default )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
end

printEvery  = setOpts( 'printEvery', 50 );
PRINT  = setOpts('verbose',1);


%% Stopping criteria
% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
target_resid    = -Inf;
if iscell(k)
    target_resid = k{1};
    k = size(b,1);
elseif k ~= round(k)
    target_resid = k;
    k   = size(b,1);
end

% (the residual is always guaranteed to decrease)
if target_resid == 0 
    if printEvery > 0 && printEvery < Inf
        disp('Warning: target_resid set to 0. This is difficult numerically: changing to 1e-12 instead');
    end
    target_resid = 1e-12;
end

if nargin < 4
    errFcn = [];   
elseif ~isempty(errFcn) && ~isa(errFcn,'function_handle')
    error('errFcn input must be a function handle (or leave the input empty)');
end

%% Construct Fourier and spike dictionary
Ab  = @(x) sqrt(n)*ifft(x(1:n)) + x(n+1:end);   % A*x
Af  = @(x) [fft(x)/sqrt(n); x];                 % whole dictionary

% -- Initialize --
% start at x = 0, so r = b - A*x = b
r           = b;                % residual
Ar          = Af(r);            % compute correlations
N           = 2*n;              % number of atoms
M           = size(r,1);        % size of atoms
if k > M
    error('K cannot be larger than the dimension of the atoms');
end

unitVector  = zeros(N,1);
x           = zeros(N,1);
xprev       = zeros(N,1);
indx_set    = zeros(k,1);
A_S         = zeros(M,k);
A_Sr        = zeros(M,k);

%% Start threshold
%  could vectorise this but that involves sorting the entire vector,
%  which could be more time consuming for large N
ICprev = Inf;

for kk = 1:k
    % -- Step 1: find new index and atom to add
    [~,ind_new] = max(abs(Ar));
    unitVector(ind_new) = 1;
    if ind_new <= n
        atom_new = fft(unitVector(1:n))/sqrt(n);
    else
        atom_new = unitVector(n+1:end);
    end
    unitVector(ind_new) = 0;     
    
    % Update support set
    indx_set(kk) = ind_new;
    A_S(:,kk) = atom_new; % insert new atom
    
    % simple greedy minimisation of information criteria
    if ~isempty(crit)
        p = 2*kk+1;        
        x(indx_set(1:kk)) = conj(A_S(:,1:kk))'*b;
        
        % updating residual based on orthogonality
        % First, orthogonalize 'atom_new' against all previous atoms
        % Modified Gram-Schmidt procedure for orthogonalization

        for j = 1:(kk-1)
            atom_new = atom_new - ((atom_new)'*A_Sr(:,j))*A_Sr(:,j);
        end
        % Second, normalize:
        atom_new  = atom_new/norm(atom_new);
        A_Sr(:,kk) = atom_new;
        
        z = zeros(N,1);
        z(indx_set(1:kk)) = (A_Sr(:,1:kk))'*b;
        r = b - A_Sr(:,1:kk)*((A_Sr(:,1:kk))'*b);

        if strcmp(crit,'aic')
            IC = 2*p + n*(2*log(norm(r))-log(n)+1);
        elseif strcmp(crit,'aicc')
            IC = 2*p + n*(2*log(norm(r))-log(n)+1) + ...
                    2*p*(p+1)/(n-p-1);
        elseif strcmp(crit,'bic')
            IC = p*log(n)+ n*(2*log(norm(r))-log(n)+1);
        elseif strcmp(crit,'mdl')
            IC = 0.5*p*log(n)+ n*log(norm(r));
        elseif strcmp(crit,'nmdl')
            RSS = norm(r)^2;
            S = RSS/(n-p);
            F = (norm(b)^2 - RSS)/(p*S);
         	IC = 0.5*n*log(S) + 0.5*p*log(F) + 0.5*log(n-p) -1.5*log(p);
        else
           error('Not one of the information criteria');        
        end       
        
        % greedy style search
        if IC >= ICprev
            x = xprev;
            break;
        end
        ICprev = IC;
        xprev = x;
    end     
    
    Ar(ind_new) = 0;
end

fprintf('Iter,  Resid,   Error\n');
if isempty(crit)
    x(indx_set(1:kk)) = conj(A_S(:,1:kk))'*b;
end
r = b - Ab(x);
bapp = real(Ab(x)); % reconstruct sparse signal

if ~isempty(errFcn)
    errHist  = errFcn(x);
    if PRINT, fprintf('%4d, %.2e, %.2e\n', 1, norm(r), errHist ); end
else
    if PRINT, fprintf('%4d, %.2e\n', 1, norm(r) ); end
end

end % end of main function