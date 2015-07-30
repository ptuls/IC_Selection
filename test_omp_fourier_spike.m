% TEST_OMP_FOURIER_SPIKE
% Test the Orthogonal Matching Pursuit and Threshold methods on an example
% signal with a single frequency and a single spike. Noise can be added by
% setting NOISE_STRENGTH to 0 (no noise), 1 (mild noise) or 2 (strong
% noise).
%
% Five information criterion are available:
%     1. 'aic': Akaike information criterion
%     2. 'aicc': Akaike information criterion corrected for small
%              samples
%     3. 'bic': Bayesian information criterion
%     4. 'mdl': two-stage Minimum Description Length
%     5. 'nmdl': normalized Minimum Description Length
% 
% Criterion is computed greedily, so choice may be suboptimal.
%
% Note: because the information criteria only approximates the signal the
% target residue is generally not reached.

% file:     test_omp_fourier_spike.m, (c) Paul Tune, Jul 03 2015
% created: 	Fri Jul 03 2015 
% author:  	Paul Tune 
% email:   	paul.tune@adelaide.edu.au

clear;

N = 1000;
K = 100;    % sparsity
L = 10;
t = linspace(0,L,N);

%% Construct signal
% construct spike
s = zeros(N,1);
s(41) = 50;     % spike at location 41

frequency = 2;
y = 10*cos(2*pi*frequency*t)'+s;

NOISE_STRENGTH = 1;

if NOISE_STRENGTH == 1
    y = y + 1*randn(N,1);
elseif NOISE_STRENGTH == 2
    y = y + 10*randn(N,1);
else
    % do nothing
end

% set information criterion
criterion = 'nmdl';

fprintf('OMP - Fourier + Spike\n');
[xft,rft,yft] = omp_fourier_spike(y,K,criterion);
fprintf('\nOMP - Fourier\n');
[xf,rf,yf] = omp_fourier(y,K,criterion);
fprintf('\nThreshold - Fourier + Spike\n');
[xt,rt,yt] = threshold_fourier_spike(y,K,criterion);

%% Plot results
dgrey = [0.65 0.65 0.65];
fontsize = 18;
thick_line = 2;

figure(1)
h1 = plot(t,y,'color',dgrey,'linewidth',thick_line);
hold on
h2 = plot(t,yft,'r','linewidth',thick_line-0.5);
h3 = plot(t,yf,'b');
h4 = plot(t,yt,'k');
hold off
set(gca,'fontsize', fontsize);
hlegend = legend([h1,h2,h3,h4],'Signal','OMP (F+S)', 'OMP (F)', ...
    'TH');
xlabel('t (s)');
ylabel('Magnitude');
title('Time domain');


figure(2)
% frequency axis for plotting
sampling_rate = N/L;
f = (0:N-1)/N*sampling_rate;

fy = abs(fft(y-s)/sqrt(N));
plot(f(1:floor(N/4)),fy(1:floor(N/4)),'color',dgrey,'linewidth',thick_line);
hold on
plot(f(1:floor(N/4)),abs(xft(1:floor(N/4))),'r','linewidth',thick_line-0.5);
plot(f(1:floor(N/4)),abs(xf(1:floor(N/4))),'b');
plot(f(1:floor(N/4)),abs(xt(1:floor(N/4))),'k');
hold off
set(gca,'fontsize', fontsize);
xlabel('f (Hz)');
ylabel('Magnitude');
title('Frequency domain');

fprintf('\nCoefficients chosen:\n');
fprintf('OMP (F+S) - Fourier: %d, Spike: %d\n', sum(abs(xft(1:N)) > 0), ...
    sum(abs(xft(N+1:end)) > 0));
fprintf('OMP (F) - Fourier: %d\n', sum(abs(xf(1:N)) > 0));
fprintf('Threshold (F+S) - Fourier: %d, Spike: %d\n', ...
    sum(abs(xt(1:N)) > 0), ...
    sum(abs(xt(N+1:end)) > 0));