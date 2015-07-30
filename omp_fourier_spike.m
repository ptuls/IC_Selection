function [x,r,bapprox,IC,residHist,errHist] = omp_fourier_spike(b,k,crit,errFcn,opts)
%OMP_FOURIER_SPIKE
%   uses the Orthogonal Matching Pursuit algorithm (OMP) to select Fourier
%   coefficients to estimate the solution to the equation
%
%       b = A*x     (or b = A*x + noise)
%
%   where there is prior information that x is sparse and A is the
%   dictionary of the Fourier and spikes. In Matlab, this is equivalent to
%
%       A = [fft(eye(N))/sqrt(N) eye(N)], where N is the length of b 
%
%   but this can be implemented more efficiently via the Fast Fourier 
%   Transform (FFT). The first N atoms come from the Fourier  
%   basis, while the next N comes from the canonical Euclidean basis. 
%   Here, note that x is twice the length of b, i.e. 2N.
%
%   x = omp_fourier_spike(b,k) find the k-sparse estimate of the unknown signal
%   base on observation b.
% 
%   [x,r,bapprox,IC,residHist,errHist] = omp_fourier(b,k,crit,errFcn,opts)
%   is the full version.
%
% Outputs:
%
%   'x' is the k-sparse estimate of the unknown signal
%   'r' is the residual b - A*x
%   'IC' minimum value of the information criterion selected (empty if 
%        no criterion was selected)
%   'normR' = norm(r)
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
%   'errFcn'    (optional; set to [] to ignore) is a function handle
%              which will be used to calculate the error; the output
%              should be a scalar
%
%   'opts'  is a structure with more options, including:
%       .printEvery = is an integer which controls how often output is printed
%       .maxiter    = maximum number of iterations
%       .slowMode   = whether to compute an estimate at every iteration
%                       This computation is slower, but it allows you to
%                       display the error at every iteration (via 'errFcn')
%
%       Note that these field names are case sensitive!
%

% file:     omp_fourier_spike.m, (c) Paul Tune, Jul 03 2015
% created: 	Fri Jul 03 2015 
% author:  	Paul Tune 
% email:   	paul.tune@adelaide.edu.au

% Modified version of OMP code by 
% Stephen Becker, Aug 1 2011.  srbecker@alumni.caltech.edu

%% Check options
if nargin < 3
    crit = []; % no information criterion used
    IC = [];
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
n = length(b);
Ab  = @(x) sqrt(n)*ifft(x(1:n)) + x(n+1:end);   % A*x
Af  = @(x) fft(x)/sqrt(n);                      % only frequencies
At  = @(x) x;                                   % only spikes
Atf = @(x) [fft(x)/sqrt(n); x];

% -- Initialize --
% start at x = 0, so r = b - A*x = b
r           = b;
normR       = norm(r);
Arf          = Af(r);
Art          = At(r);
Ar          = Atf(r);
N           = 2*n;       % number of atoms
M           = size(r,1);        % size of atoms
if k > M
    error('K cannot be larger than the dimension of the atoms');
end
unitVector  = zeros(N,1);
x           = zeros(N,1);
xprev       = zeros(N,1);

indx_set    = zeros(k,1);
A_S         = zeros(M,k);
A_S_nonorth = zeros(M,k);
residHist   = zeros(k,1);
errHist     = zeros(k,1);

% count atoms
freq_atom = zeros(k,1);
spike_atom = zeros(k,1);

ICprev = Inf;
%% Start matching pursuit
fprintf('Iter,  Resid,   Error\n');
kk = 1;
for kk = 1:k
% while (kk < k)
    % -- Step 1: find new index and atom to add
    [~,ind_new] = max(abs(Ar));  
    indx_set(kk) = ind_new;   
    unitVector(ind_new)     = 1;
    
    % faster updating
    if (ind_new > n) % divide into spikes and Fourier
        atom_new = unitVector(n+1:end);
        spike_atom(kk) = 1;
    else
        atom_new = fft(unitVector(1:n))/sqrt(n);                
        freq_atom(kk) = 1;
    end
    A_S_nonorth(:,kk) = atom_new;     % before orthogonalizing and such
    unitVector(ind_new) = 0; % reset

    
    % -- Step 2: update residual
    % First, orthogonalize 'atom_new' against all previous atoms
    % Modified Gram-Schmidt procedure for orthogonalization
    for j = 1:(kk-1)
        atom_new = atom_new - dot(A_S(:,j),A_S_nonorth(:,kk))*A_S(:,j);
    end
    % Second, normalize:
    atom_new  = atom_new/norm(atom_new);
    A_S(:,kk) = atom_new; % insert new atom       

    % Third, solve least-squares problem (which is now very easy
    %   since A_S(:,1:kk) is orthogonal )
    x_S = conj(A_S(:,1:kk))'*b;     % conj due to complex dot product
    x(indx_set(1:kk)) = x_S;  % note: indx_set is guaranteed to never shrink
    % Fourth, update residual:
    %      r = b - Ab(x); % wrong!
    % need to remove the frequency part, then the time spikes    
    r = b - conj(A_S(:,1:kk))*x_S;  % remember IFFT

    if ~isempty(crit)
        p = 2*kk+1;
        bapprox = real(sqrt(n)*ifft(x(1:n)));
        id = find(abs(x(n+1:end)) > 0);
        bapprox(id) = x(n+id);
        rt = b - bapprox;
        
        if strcmp(crit,'aic')
            IC = 2*p + n*(2*log(norm(rt))-log(n)+1);
        elseif strcmp(crit,'aicc')
            IC = 2*p + n*(2*log(norm(rt))-log(n)+1) + ...
                    2*p*(p+1)/(n-p-1);
        elseif strcmp(crit,'bic')
            IC = p*log(n)+ n*(2*log(norm(rt))-log(n)+1);
        elseif strcmp(crit,'mdl')
            IC = 0.5*p*log(n)+ n*log(norm(rt));
        elseif strcmp(crit,'nmdl')
            IC = 0.5*p*log(norm(b).^2 - norm(rt)^2)-0.5*(n-p-1)*log(n-p)...
                    - 0.5*(p+3)*log(p) + (n-p)*log(norm(rt));
        else
           error('Not one of the information criteria');        
        end       

        % may potentially be stuck in local minimum
        if IC > ICprev
            x = xprev;
            break;                
        end
        ICprev = IC;
        xprev = x;
    end

    % N.B. This err is unreliable, since this "x" is not the same
    %   (since it relies on A_S, which is the orthogonalized version)
    
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
    
    if kk < k
        Ar = Atf(r);        
    end
end

if (target_resid) && ( normR >= target_resid )
    fprintf('Warning: did not reach target size of residual\n');
end

% For the last iteration, we need to do this without orthogonalizing A
% so that the x coefficients match what is expected.
% no inversion needed: Fourier 
% kk = kk-1;
% disp(indx_set)
x(indx_set(1:kk)) = conj(A_S_nonorth(:,1:kk))'*b;
% r = b - Ab(x);
normR = norm(r);
% reconstruct sparse signal
bapprox = real(sqrt(n)*ifft(x(1:n)));
id = find(abs(x(n+1:end)) > 0);
bapprox(id) = x(n+id);

end % end of main function

