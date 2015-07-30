function [x,r,bapprox,IC,residHist,errHist] = omp_fourier(b,k,crit,errFcn,opts)
%OMP_FOURIER 
%   uses the Orthogonal Matching Pursuit algorithm (OMP) to select Fourier
%   coefficients to estimate the solution to the equation
%
%       b = A*x     (or b = A*x + noise)
%
%   where there is prior information that x is sparse and A is the
%   dictionary of the Fourier and spikes. In Matlab, this is equivalent to
%
%       A = fft(eye(N))/sqrt(N), where N is the length of x 
%
%   but this can be implemented more efficiently via the Fast Fourier 
%   Transform (FFT). 
%
%   x = omp_fourier(b,k) find the k-sparse estimate of the unknown signal
%   base on observation b.
% 
%   [x,r,bapprox,IC,residHist,errHist] = omp_fourier(b,k,crit,errFcn,opts)
%   is the full version.
%
% Outputs:
%
%   'x' is the k-sparse estimate of the unknown signal
%   'r' is the residual b - A*x
%   'bapprox' is the reconstructed signal approximation of b
%   'IC' minimum value of the information criterion selected (empty if 
%        no criterion was selected)
%   'residHist' is a vector with normR from every iteration
%   'errHist' is a vector with the outout of errFcn from every iteration
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
%           criterion used. Five are available:
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

% file:     omp_fourier.m, (c) Paul Tune, Jul 03 2015
% created: 	Fri Jul 03 2015 
% author:  	Paul Tune 
% email:   	paul.tune@adelaide.edu.au

% Modified version from OMP code by 
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

printEvery  = setOpts('printEvery',50);

%% Stopping criteria
% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
target_resid    = -Inf;
if iscell(k)
    target_resid = k{1};
    k = size(b,1);
elseif k ~= round(k)
    target_resid = k;
    k = size(b,1);
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

%% Construct Fourier dictionary
At = @(x) sqrt(n)*ifft(x(1:n));   % A*x (unnormalized)
Af = @(x) fft(x)/sqrt(n);         % conj(A)'*x (normalized)

% -- Initialize --
% start at x = 0, so r = b - A*x = b
r           = b;                % residue
normR       = norm(r);          % norm of residue
Ar          = Af(r);            % compute correlations
N           = size(Ar,1);       % number of atoms
M           = size(r,1);        % size of atoms

if k > M
    error('k cannot be larger than the dimension of the atoms');
end
unitVector  = zeros(N,1);
x           = zeros(N,1);
xprev       = zeros(N,1);

indx_set    = zeros(k,1);
indx_set_sorted     = zeros(k,1);
A_S = zeros(M,k);   % support vectors: Fourier basis
residHist   = zeros(k,1);
errHist     = zeros(k,1);

ICprev = Inf;       % initial information criterion value

%% Start orthogonal matching pursuit
fprintf('Iter,  Resid,   Error\n');
for kk = 1:k
    % -- Step 1: find new index and atom to add
    [~,ind_new] = max(abs(Ar));
       
    indx_set(kk) = ind_new;
    indx_set_sorted(1:kk) = sort(indx_set(1:kk));
    
    % remember: the atoms are the Fourier basis
    unitVector(ind_new) = 1;
    atom_new = Af(unitVector);  % Fourier transform        
    % don't need an orthogonalizing step since Fourier atoms are orthogonal
    % to each other
    A_S(:,kk) = atom_new;     
    unitVector(ind_new) = 0; % reset
    
    % -- Step 2: update residual and compute information criterion
    x_S = conj(A_S(:,1:kk))'*b; % no inversion needed: Fourier orthogonal    
    x(indx_set(1:kk)) = x_S; % update support
    
    % remove contribution: can do this because of orthogonality of
    % dictionary atoms (could use IFFT instead)
    r = b - conj(A_S(:,1:kk))*x_S;

    if ~isempty(crit)
        p = 2*kk+1;
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
        
        % may potentially be stuck in local minimum
        if IC >= ICprev
            x = xprev;
            break;
        end
        ICprev = IC;
        xprev = x;
    end
    
    %% Print statistics
    normR   = norm(r);
    % -- Print some info --
    PRINT   = ( ~mod( kk, printEvery ) || kk == k );
    if printEvery > 0 && printEvery < Inf && (normR < target_resid )
        % this is our final iteration, so display info
        PRINT = true;
    end

    if ~isempty(errFcn)
        er  = errFcn(x);
        if PRINT, fprintf('%4d, %.2e, %.2e\n', kk, normR, er ); end
        errHist(kk) = er;
    else
        if PRINT, fprintf('%4d, %.2e\n', kk, normR ); end
    end
    residHist(kk) = normR;
    
    if normR < target_resid
        if PRINT
            fprintf('Residual reached desired size (%.2e < %.2e)\n', normR, target_resid );
        end
        break;
    end

    %% Prepare for next round
    if kk < k
        Ar  = Af(r); 
    end
end

if (target_resid) && ( normR >= target_resid )
    fprintf('Warning: did not reach target size of residual\n');
end

bapprox = real(At(x)); % reconstruct sparse signal
end % end of main function

