function out = IMPULSED_bayes(Y,f,diameter,Dex,seqParams,lim,n,prior,burns,E_set,D_set)
% out = IMPULSED_bayes(Y,f,diameter,Dex,seqParams,lim,n,prior,burns,E_set,D_set)
%
% Input: 
%       Y :  An N_PULSE - dimensional column vector containing signal values of an individual voxel 
%            across N_PULSE measurement sequences.
%       f :  initial value of parameter fin
%       diameter :  initial value of parameter diameter (μm)
%       Dex : initial value of parameter Dex (μm²/ms)
%       seqParams :  a N_PULSE x6 vector containing [smallDelta, bigDelta, freq, Grad, bval,ramptime]
%       lim : a 2x3 matrix with lower (1st row) and upper (2nd row)  limits of all parameters 
%             in the order f,diameter,Dex
%       n :  number of iterations after "burn in" (default: 5000)
%       prior : A cell variable of size 1x3, sequentially specifying the prior distribution functions for 
%               fin, diameter, and Dex, including 'flat', 'reci', 'Gaussian' or 'lognorm'.(default: {'flat','flat','flat'}
%              'flat' = uniform, 'reci' = reciprocal,'Gaussian' = Gaussian , 'lognorm' = Logrithm Gaussian
%       burns :  number of burn-in steps (default: 5000)
%       E_set: The expectation under Gaussian or logarithmic Gaussian priors, 
%              i.e., the IMPULSED model fitting results under the NLLS framework.
%       D_set: The variance values under Gaussian or logarithmic Gaussian priors, 


% Output:
%       out: a struct with the fields D, f, Dstar and S0(Vx1) containing the
%       voxelwise mean, median, mode and standard deviation of each parameter
%


% N = number of iterations

if nargin < 7
    n = 5000;
else
    if ~(isscalar(n) && n > 0 && (round(n) == n) && isreal(n))
        error('N must be a positive scalar integer');
    end
end
        
if nargin < 8
    prior = {'flat','flat','flat'}; % use flat prior distributions
end

% burn-in steps
if nargin < 9
    burns = 5000;
end

burnUpdateInterval = 100;
burnUpdateFraction = 1/2;
%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter estimation %
%%%%%%%%%%%%%%%%%%%%%%%%

% initialize parameter vector
theta = zeros(1,3,n+burns);
theta(:,1:3,1) = [f, diameter, Dex];

% step length parameter
w = zeros(1,3);
w(:,1:3) = [f diameter Dex]/10;

N = zeros(1,3); % number of accepted samples

% iterate for j = 2,3,...,n
for j = 2:n + burns
    % initialize theta(j)
      theta(:,:,j) = theta(:,:,j-1);
      thetanew = theta(:,:,j);
      thetaold = theta(:,:,j-1);
      
    % sample each parameter
    for k = 1:3
        % sample s and r and update
        s = thetaold(:,k) + randn.*w(:,k);
        r = rand; 

        thetas = thetanew;
        thetas(:,k) = s;
        alpha = acc_MH(thetas,thetanew,Y,seqParams,lim,prior{k},E_set,D_set);
        sample_ok = r < alpha;
        thetanew(sample_ok,k) = thetas(sample_ok,k);
        thetanew(~sample_ok,k) = thetaold(~sample_ok,k); % reject samples
        N(:,k) = N(:,k) + sample_ok;
    end

    % prepare for next iteration
    theta(:,:,j) = thetanew;

    % adapt step length
    if j <= burns*burnUpdateFraction && mod(j,burnUpdateInterval) == 0
        w = w*(burnUpdateInterval+1)./(2*((burnUpdateInterval+1)-N));
        N = zeros(1,3);
    end

    % Display iteration every 500th iteration
    if ~mod(j,500) && j > burns
        disp(['Iterations: ' num2str(j-burns)]);
    elseif ~mod(j,100) && j < burns
        disp(['Burn in-steps: ' num2str(j)]);
    elseif j == burns
        disp(['Burn in complete: ' num2str(j)]);
    end
end

% Saves distribution measures

%mean
out.f.mean = mean(squeeze(theta(:,1,burns + 1:n+burns)),1);
out.diameter.mean = mean(squeeze(theta(:,2,burns + 1:n+burns)),1);
out.Dex.mean = mean(squeeze(theta(:,3,burns + 1:n+burns)),1);

%median
out.f.median = median(squeeze(theta(:,1,burns + 1:n+burns)),1);
out.diameter.median = median(squeeze(theta(:,2,burns + 1:n+burns)),1);
out.Dex.median = median(squeeze(theta(:,3,burns + 1:n+burns)),1);

%mode
out.f.mode = halfSampleMode(squeeze(theta(:,1,burns + 1:n+burns))');
out.diameter.mode = halfSampleMode(squeeze(theta(:,2,burns + 1:n+burns))');
out.Dex.mode = halfSampleMode(squeeze(theta(:,3,burns + 1:n+burns))');

% standard deviation
out.f.std = std(squeeze(theta(:,1,burns + 1:n+burns)),1,1);
out.diameter.std = std(squeeze(theta(:,2,burns + 1:n+burns)),1,1);
out.Dex.std = std(squeeze(theta(:,3,burns + 1:n+burns)),1,1);




function alpha = acc_MH(thetas,thetaj,Y,seqParams,lim,prior,E_set,D_set)
% theta = [f, D, Dstar,S0,1/s2];
M = size(thetas,1);
N = size(seqParams,1);

q = zeros(M,1);

% p(theta|lim)
pts = min((thetas(:,1:3) >= repmat(lim(1,:),M,1)) & (thetas(:,1:3) <= repmat(lim(2,:),M,1)),[],2);

% % D < D*
% pts = pts & (thetas(:,2) < thetas(:,3));

% signal model 
Ss = zeros(M,N);
Sj = Ss;
for voxel = 1:M
    if pts(voxel) == true
        Ss(voxel,:) = IMPULSED_fixDin_3([thetas(voxel,1),thetas(voxel,2),thetas(voxel,3)],seqParams);
        Sj(voxel,:) = IMPULSED_fixDin_3([thetaj(voxel,1),thetaj(voxel,2),thetaj(voxel,3)],seqParams);
    end
end
ptsptj = double(pts);

if strcmp(prior,'reci')
    diffpar = find(thetas(1,:) ~= thetaj(1,:));
    ptsptj(pts) = thetaj(pts,diffpar)./thetas(pts,diffpar); % rejects samples outside the limits % ~pts already == 0

elseif strcmp(prior,'Gaussian')
    diffpar = find(thetas(1,:) ~= thetaj(1,:));
    
%     ptsptj = pts;
    if diffpar == 2         % diameter
        mu = E_set;
        s = D_set;
    else
        error('norm prior not available'); 
    end
    ptsptj(pts) = normprior(thetas(pts,diffpar),mu,s)./normprior(thetaj(pts,diffpar),mu,s); % ~pts already == 0
elseif strcmp(prior,'Lognorm')
    diffpar = find(thetas(1,:) ~= thetaj(1,:));
    
    if diffpar == 2         % diameter
        Ex = E_set;
        Dx = D_set;
        
        mu = log(Ex)-log(1+Dx/Ex^2)/2;
        s = sqrt(log(1+Dx/Ex^2));   
    else
        error('lognorm prior not available'); % only for diameter
    end
    ptsptj(pts) = lognormprior(thetas(pts,diffpar),mu,s)./lognormprior(thetaj(pts,diffpar),mu,s); % ~pts already == 0  
elseif ~strcmp(prior,'flat')
    error('unknown prior');
end

q(pts) = (sum((Y(pts,:)-Sj(pts,:)).^2,2)./sum((Y(pts,:)-Ss(pts,:)).^2,2)).^(N/2) .* ptsptj(pts);


alpha = min(1  ,q);


function p = normprior(x,mu,s)
p = 1./(s*sqrt(2*pi)).*exp(-(x-mu).^2/(2*s^2));


function p = lognormprior(x,mu,s)
p = 1./(s*sqrt(2*pi)*x).*exp(-(log(x)-mu).^2/(2*s^2));

