function S = IMPULSED_fixDin_3(x, seqParams)
% x = [fin, diameter, Dex];
%       x(1)  x(2)   x(3) 
% seqParams = [smallDelta, bigDelta, freq, Grad, bval,ramptime];
% grad in unit of T/m
% diameter in unit of meter
% Din, Dex in unit of m^2/s
% smallDelta and bigDelta in unit of second
K = 10;
%Din = 1e-9;
x = [x(1), x(2)*1e-6, x(3)*1e-9];
Din=1e-9;

S = zeros(size(seqParams, 1), 1);
for k = 1:size(seqParams, 1)
    N = round(seqParams(k,1)*seqParams(k,3));
    ramptime = seqParams(k,6);
    if N==1
    %     bval = bvalue_trapOGSE(freq, smallDelta, 0.15, Grad);
        Sin = sig_OGSE1(x(2), Din, seqParams(k,1), seqParams(k,2), seqParams(k,3), seqParams(k,4),ramptime, K);
    elseif N==2
        
        Sin = sig_OGSE2(x(2), Din, seqParams(k,1), seqParams(k,2), seqParams(k,3), seqParams(k,4),ramptime, K);
    elseif N==3
        
        Sin = sig_OGSE3(x(2), Din, seqParams(k,1), seqParams(k,2), seqParams(k,3), seqParams(k,4),ramptime, K);
    elseif N==0
    %     bval = bvalue_PGSE(smallDelta, bigDelta, 0.15, Grad);
        Sin = sig_PGSE(x(2),Din, seqParams(k,1), seqParams(k,2), seqParams(k,4),ramptime,K);     
    end
    
    Sex = exp(-seqParams(k,5)*x(3));
    S(k) = x(1)*Sin + (1-x(1))*Sex;
end


% Sin for PGSE
function S = sig_PGSE(diameter, Din, smallDelta, bigDelta, Grad,ramptime, K)
% By Li Hua
% gdiff = gdiff.*1e2; delta = delta.*1e-3; Delta = Delta.*1e-3; D = D.*1e-9; radius = radius.*1e-6;
% gamma_1 = 267.5e6;
gamma = 267.513e6; % rad/(s*T)

tmp = 0;
for k = 1:K
    [Bk, lambda_k] = calcBk(diameter, k);
    tt = lambda_k*Din;
    tr=ramptime;
    tp=smallDelta-2*tr;
%     tmp = tmp+2*Bk*gamma^2*Grad^2/tt^2*(tt*smallDelta-1 + exp(-tt*smallDelta))...
%         + exp(-tt*bigDelta)*(1-cosh(tt*smallDelta));
% Break cosh(x) into (exp(x)+exp(-x))/2, according to Li Hua
    tmp = tmp+ Bk/(tt^4*tr^2)*(2*exp(-tt*tp)-4*exp(-tt*tr)-4*exp(-bigDelta*tt)-4*tt*tr+2*exp(-tt*(bigDelta-tr))+2*exp(-tt*(bigDelta+tr))...
        -exp(-tt*(bigDelta-tp))-exp(-tt*(bigDelta+tp))-4*exp(-tt*(tr+tp))+2*exp(-tt*(2*tr+tp))+4/3*tt^3*tr^3+2*tt^3*tr^2*tp...
        +2*exp(-tt*(bigDelta-tr-tp))+2*exp(-tt*(bigDelta+tr+tp))-exp(-tt*(bigDelta-2*tr-tp))-exp(-tt*(bigDelta+2*tr+tp))+4);
end

S = exp(-gamma^2*Grad^2*tmp);

% Sin for OGSE
function S = sig_OGSE1(diameter, Din, smallDelta, bigDelta, freq, Grad,ramptime, K)
gamma = 267.513e6; % 1e6*rad/(s*T)
tmp = 0;
for k = 1:K
    [Bk, lambda_k] = calcBk(diameter, k);
    tt = lambda_k*Din;
    tr=ramptime;
    tp=(smallDelta/2-tr*3.5)/2;
    tmp = tmp+ Bk/(tt^4*tr^2)*(4*exp(-tt*tp)-4*exp(-tt*tr)-4*exp(-2*tt*tr)-8*exp(-tt*bigDelta)-12*tt*tr+2*exp(-tt*(bigDelta-tr))...
        +2*exp(-tt*(bigDelta+tr))+2*exp(-tt*(bigDelta+2*tr))+2*exp(-tt*(bigDelta-2*tr))-2*exp(-tt*(bigDelta-tp))-2*exp(-tt*(bigDelta+tp))...
        -4*exp(-tt*(tr+tp))+2*exp(-tt*(tr+2*tp))-4*exp(-tt*(2*tr+tp))+4*exp(-tt*(3*tr+tp))-4*exp(-tt*(3*tr+2*tp))-4*exp(-tt*(3*tr+3*tp))...
        +4*exp(-tt*(4*tr+3*tp))+2*exp(-tt*(5*tr+2*tp))+4*exp(-tt*(5*tr+3*tp))+2*exp(-tt*(5*tr+4*tp))-4*exp(-tt*(6*tr+3*tp))-4*exp(-tt*(6*tr+4*tp))...
        +2*exp(-tt*(7*tr+4*tp))+6*tt^3*tr^3+8*tt^3*tr^2*tp+2*exp(-tt*(bigDelta-tr-tp))-exp(-tt*(bigDelta-tr-2*tp))+2*exp(-tt*(bigDelta+tr+tp))...
        +2*exp(-tt*(bigDelta-2*tr-tp))-exp(-tt*(bigDelta+tr+2*tp))+2*exp(-tt*(bigDelta+2*tr+tp))-2*exp(-tt*(bigDelta-3*tr-tp))-2*exp(-tt*(bigDelta+3*tr+tp))...
        +2*exp(-tt*(bigDelta+3*tr+2*tp))+2*exp(-tt*(bigDelta-3*tr-2*tp))+2*exp(-tt*(bigDelta+3*tr+3*tp))+2*exp(-tt*(bigDelta-3*tr-3*tp))...
        -2*exp(-tt*(bigDelta+4*tr+3*tp))-2*exp(-tt*(bigDelta-4*tr-3*tp))-exp(-tt*(bigDelta+5*tr+2*tp))-exp(-tt*(bigDelta-5*tr-2*tp))...
        -2*exp(-tt*(bigDelta+5*tr+3*tp))-2*exp(-tt*(bigDelta-5*tr-3*tp))-exp(-tt*(bigDelta+5*tr+4*tp))-exp(-tt*(bigDelta-5*tr-4*tp))...
        +2*exp(-tt*(bigDelta+6*tr+3*tp))+2*exp(-tt*(bigDelta-6*tr-3*tp))+2*exp(-tt*(bigDelta+6*tr+4*tp))+2*exp(-tt*(bigDelta-6*tr-4*tp))...
        -exp(-tt*(bigDelta+7*tr+4*tp))-exp(-tt*(bigDelta-7*tr-4*tp))+8);
end

S = exp(-gamma^2*Grad^2*tmp);

function S = sig_OGSE2(diameter, Din, smallDelta, bigDelta, freq, Grad,ramptime, K)
gamma = 267.513e6; % 1e6*rad/(s*T)

tmp = 0;
for k = 1:K
    [Bk, lambda_k] = calcBk(diameter, k);
    tt = lambda_k*Din;
    tr=ramptime;
    tp=(smallDelta/2-tr*6.5)/4;
    tmp = tmp+Bk/(tt^4*tr^2)*(4*exp(-tt*tp)-4*exp(-tt*tr)-8*exp(-2*tt*tr)-12*exp(-tt*bigDelta)-20*tt*tr+2*exp(-tt*(bigDelta-tr))...
        +2*exp(-tt*(bigDelta+tr))+4*exp(-tt*(bigDelta+2*tr))+4*exp(-tt*(bigDelta-2*tr))-2*exp(-tt*(bigDelta-tp))...
        -2*exp(-tt*(bigDelta+tp))-4*exp(-tt*(tr+tp))+6*exp(-tt*(tr+2*tp))-4*exp(-tt*(2*tr+tp))...
        +4*exp(-tt*(3*tr+tp))-12*exp(-tt*(3*tr+2*tp))-4*exp(-tt*(3*tr+3*tp))+4*exp(-tt*(4*tr+3*tp))...
        +6*exp(-tt*(5*tr+2*tp))-4*exp(-tt*(4*tr+4*tp))+4*exp(-tt*(5*tr+3*tp))-4*exp(-tt*(6*tr+3*tp))...
        +8*exp(-tt*(6*tr+4*tp))+4*exp(-tt*(6*tr+5*tp))-4*exp(-tt*(7*tr+5*tp))-4*exp(-tt*(8*tr+4*tp))...
        +2*exp(-tt*(7*tr+6*tp))-4*exp(-tt*(8*tr+5*tp))+4*exp(-tt*(9*tr+5*tp))-4*exp(-tt*(9*tr+6*tp))...
        -4*exp(-tt*(9*tr+7*tp))+4*exp(-tt*(10*tr+7*tp))+2*exp(-tt*(11*tr+6*tp))+4*exp(-tt*(11*tr+7*tp))...
        +2*exp(-tt*(11*tr+8*tp))-4*exp(-tt*(12*tr+7*tp))-4*exp(-tt*(12*tr+8*tp))+2*exp(-tt*(13*tr+8*tp))...
        +38/3*tt^3*tr^3+16*tt^3*tr^2*tp+2*exp(-tt*(bigDelta-tr-tp))-3*exp(-tt*(bigDelta-tr-2*tp))...
        +2*exp(-tt*(bigDelta+tr+tp))+2*exp(-tt*(bigDelta-2*tr-tp))-3*exp(-tt*(bigDelta+tr+2*tp))...
        +2*exp(-tt*(bigDelta+2*tr+tp))-2*exp(-tt*(bigDelta-3*tr-tp))-2*exp(-tt*(bigDelta+3*tr+tp))...
        +6*exp(-tt*(bigDelta+3*tr+2*tp))+6*exp(-tt*(bigDelta-3*tr-2*tp))+2*exp(-tt*(bigDelta+3*tr+3*tp))...
        +2*exp(-tt*(bigDelta-3*tr-3*tp))-2*exp(-tt*(bigDelta+4*tr+3*tp)) -2*exp(-tt*(bigDelta-4*tr-3*tp))...
        -3*exp(-tt*(bigDelta+5*tr+2*tp))-3*exp(-tt*(bigDelta-5*tr-2*tp))+2*exp(-tt*(bigDelta+4*tr+4*tp))...
        +2*exp(-tt*(bigDelta-4*tr-4*tp))-2*exp(-tt*(bigDelta+5*tr+3*tp))-2*exp(-tt*(bigDelta-5*tr-3*tp))...
        +2*exp(-tt*(bigDelta+6*tr+3*tp))+2*exp(-tt*(bigDelta-6*tr-3*tp))-4*exp(-tt*(bigDelta+6*tr+4*tp))...
        -4*exp(-tt*(bigDelta-6*tr-4*tp))-2*exp(-tt*(bigDelta+6*tr+5*tp))-2*exp(-tt*(bigDelta-6*tr-5*tp))...
        +2*exp(-tt*(bigDelta+7*tr+5*tp))+2*exp(-tt*(bigDelta-7*tr-5*tp))+2*exp(-tt*(bigDelta+8*tr+4*tp))...
        +2*exp(-tt*(bigDelta-8*tr-4*tp))-exp(-tt*(bigDelta+7*tr+6*tp))-exp(-tt*(bigDelta-7*tr-6*tp))...
        +2*exp(-tt*(bigDelta+8*tr+5*tp))+2*exp(-tt*(bigDelta-8*tr-5*tp))-2*exp(-tt*(bigDelta+9*tr+5*tp))...
        -2*exp(-tt*(bigDelta-9*tr-5*tp))+2*exp(-tt*(bigDelta+9*tr+6*tp))+2*exp(-tt*(bigDelta-9*tr-6*tp))...
        +2*exp(-tt*(bigDelta+9*tr+7*tp))+2*exp(-tt*(bigDelta-9*tr-7*tp))-2*exp(-tt*(bigDelta+10*tr+7*tp))...
        -2*exp(-tt*(bigDelta-10*tr-7*tp))-exp(-tt*(bigDelta+11*tr+6*tp))-exp(-tt*(bigDelta-11*tr-6*tp))...
        -2*exp(-tt*(bigDelta+11*tr+7*tp))-2*exp(-tt*(bigDelta-11*tr-7*tp))-exp(-tt*(bigDelta+11*tr+8*tp))...
        -exp(-tt*(bigDelta-11*tr-8*tp))+2*exp(-tt*(bigDelta+12*tr+7*tp))+2*exp(-tt*(bigDelta-12*tr-7*tp))...
        +2*exp(-tt*(bigDelta+12*tr+8*tp))+2*exp(-tt*(bigDelta-12*tr-8*tp))-exp(-tt*(bigDelta+13*tr+8*tp))...
        -exp(-tt*(bigDelta-13*tr-8*tp))+12);
end

S = exp(-gamma^2*Grad^2*tmp);

function S = sig_OGSE3(diameter, Din, smallDelta, bigDelta, freq, Grad,ramptime, K)
gamma = 267.513e6; % 1e6*rad/(s*T)

tmp = 0;
for k = 1:K
    [Bk, lambda_k] = calcBk(diameter, k);
    tt = lambda_k*Din;
    tr=ramptime;
    tp=(smallDelta/2-tr*9.5)/6;
%     tmp = tmp+2*Bk/4*((48*exp(-2*tt*tr) - 72*exp(-bigDelta*tt) - 60*exp(-2*tt*tp) - 120*tt*tp - 30*exp(-bigDelta*tt)...
%         *exp(-2*tt*tr) - 30*exp((2*tr-bigDelta)*tt) + 36*exp(-bigDelta*tt)*exp(-2*tt*tp) + 36*exp((2*tp-bigDelta)*tt)...
%         - 96*exp(-2*tt*tr)*exp(-2*tt*tp) + 48*exp(-2*tt*tr)*exp(-4*tt*tp) - 36*exp(-4*tt*tr)*exp(-2*tt*tp) ...
%         + 72*exp(-4*tt*tr)*exp(-4*tt*tp) - 36*exp(-4*tt*tr)*exp(-6*tt*tp) + 24*exp(-6*tt*tr)*exp(-4*tt*tp) ...
%         - 48*exp(-6*tt*tr)*exp(-6*tt*tp) + 24*exp(-6*tt*tr)*exp(-8*tt*tp) - 12*exp(-8*tt*tr)*exp(-6*tt*tp) ...
%         + 24*exp(-8*tt*tr)*exp(-8*tt*tp) - 12*exp(-8*tt*tr)*exp(-10*tt*tp) + 6*exp(-10*tt*tr)*exp(-8*tt*tp) ...
%         - 12*exp(-10*tt*tr)*exp(-10*tt*tp) + 6*exp(-10*tt*tr)*exp(-12*tt*tp) - 12*tt.^2*tp.^2 + 40*tt.^3*tp.^3 ...
%         + 12*tt.^2*tp.^2*exp(-bigDelta*tt) + 120*tt.^3*tr*tp.^2 + 24*tt*tp*exp(-tt*tr) + 60*exp(-bigDelta*tt)*exp(-2*tt*tr)*exp(-2*tt*tp) ...
%         + 60*exp((2*tr+2*tp-bigDelta)*tt) - 30*exp(-bigDelta*tt)*exp(-2*tt*tr)*exp(-4*tt*tp) - 30*exp((2*tr+4*tp-bigDelta)*tt) ...
%         + 24*exp(-bigDelta*tt)*exp(-4*tt*tr)*exp(-2*tt*tp) + 24*exp((4*tr+2*tp-bigDelta)*tt) - 48*exp(-bigDelta*tt)*exp(-4*tt*tr)*exp(-4*tt*tp) ...
%         - 48*exp((4*tr+4*tp-bigDelta)*tt) + 24*exp(-bigDelta*tt)*exp(-4*tt*tr)*exp(-6*tt*tp) + 24*exp((4*tr+6*tp-bigDelta)*tt) ...
%         - 18*exp(-bigDelta*tt)*exp(-6*tt*tr)*exp(-4*tt*tp) - 18*exp((6*tr+4*tp-bigDelta)*tt) + 36*exp(-bigDelta*tt)*exp(-6*tt*tr)*exp(-6*tt*tp) ...
%         + 36*exp((6*tr+6*tp-bigDelta)*tt) - 18*exp(-bigDelta*tt)*exp(-6*tt*tr)*exp(-8*tt*tp) - 18*exp((6*tr+8*tp-bigDelta)*tt) ...
%         + 12*exp(-bigDelta*tt)*exp(-8*tt*tr)*exp(-6*tt*tp) + 12*exp((8*tr+6*tp-bigDelta)*tt) - 24*exp(-bigDelta*tt)*exp(-8*tt*tr)*exp(-8*tt*tp) ...
%         - 24*exp((8*tr+8*tp-bigDelta)*tt) + 12*exp(-bigDelta*tt)*exp(-8*tt*tr)*exp(-10*tt*tp) + 12*exp((8*tr+10*tp-bigDelta)*tt) ...
%         - 6*exp(-bigDelta*tt)*exp(-10*tt*tr)*exp(-8*tt*tp) - 6*exp((10*tr+8*tp-bigDelta)*tt) + 12*exp(-bigDelta*tt)*exp(-10*tt*tr)*exp(-10*tt*tp) ...
%         + 12*exp((10*tr+10*tp-bigDelta)*tt) - 6*exp(-bigDelta*tt)*exp(-10*tt*tr)*exp(-12*tt*tp) - 6*exp((10*tr+12*tp-bigDelta)*tt) ...
%         + 6*tt.^2*tp.^2*exp(-8*tt*tr)*exp(-8*tt*tp) + 6*tt.^2*tp.^2*exp(-12*tt*tr)*exp(-12*tt*tp) + 12*tt*tp*exp((tr-bigDelta)*tt) ...
%         - 12*tt*tp*exp(-bigDelta*tt)*exp(-tt*tr) - 24*tt*tp*exp(-tt*tr)*exp(-2*tt*tp) - 24*tt*tp*exp(-3*tt*tr)*exp(-2*tt*tp) ...
%         + 24*tt*tp*exp(-3*tt*tr)*exp(-4*tt*tp) + 24*tt*tp*exp(-5*tt*tr)*exp(-4*tt*tp) - 24*tt*tp*exp(-5*tt*tr)*exp(-6*tt*tp) ...
%         - 24*tt*tp*exp(-7*tt*tr)*exp(-6*tt*tp) + 24*tt*tp*exp(-7*tt*tr)*exp(-8*tt*tp) + 12*tt*tp*exp(-9*tt*tr)*exp(-8*tt*tp) ...
%         - 12*tt*tp*exp(-9*tt*tr)*exp(-10*tt*tp) - 12*tt*tp*exp(-11*tt*tr)*exp(-10*tt*tp) + 12*tt*tp*exp(-11*tt*tr)*exp(-12*tt*tp) ...
%         - 6*tt.^2*tp.^2*exp(-bigDelta*tt)*exp(-12*tt*tr)*exp(-12*tt*tp) - 6*tt.^2*tp.^2*exp((12*tr+12*tp-bigDelta)*tt) ...
%         - 12*tt*tp*exp((tr+2*tp-bigDelta)*tt) + 12*tt*tp*exp(-bigDelta*tt)*exp(-tt*tr)*exp(-2*tt*tp) ...
%         + 12*tt*tp*exp(-bigDelta*tt)*exp(-3*tt*tr)*exp(-2*tt*tp) - 12*tt*tp*exp((3*tr+2*tp-bigDelta)*tt) ...
%         - 12*tt*tp*exp(-bigDelta*tt)*exp(-3*tt*tr)*exp(-4*tt*tp) + 12*tt*tp*exp((3*tr+4*tp-bigDelta)*tt) ...
%         - 12*tt*tp*exp(-bigDelta*tt)*exp(-5*tt*tr)*exp(-4*tt*tp) + 12*tt*tp*exp((5*tr+4*tp-bigDelta)*tt) ...
%         + 12*tt*tp*exp(-bigDelta*tt)*exp(-5*tt*tr)*exp(-6*tt*tp) - 12*tt*tp*exp((5*tr+6*tp-bigDelta)*tt) ...
%         + 12*tt*tp*exp(-bigDelta*tt)*exp(-7*tt*tr)*exp(-6*tt*tp) - 12*tt*tp*exp((7*tr+6*tp-bigDelta)*tt) ...
%         - 12*tt*tp*exp(-bigDelta*tt)*exp(-7*tt*tr)*exp(-8*tt*tp) + 12*tt*tp*exp((7*tr+8*tp-bigDelta)*tt) ...
%         - 12*tt*tp*exp(-bigDelta*tt)*exp(-9*tt*tr)*exp(-8*tt*tp) + 12*tt*tp*exp((9*tr+8*tp-bigDelta)*tt) ...
%         + 12*tt*tp*exp(-bigDelta*tt)*exp(-9*tt*tr)*exp(-10*tt*tp) - 12*tt*tp*exp((9*tr+10*tp-bigDelta)*tt) ...
%         + 12*tt*tp*exp(-bigDelta*tt)*exp(-11*tt*tr)*exp(-10*tt*tp) - 12*tt*tp*exp((11*tr+10*tp-bigDelta)*tt) ...
%         - 12*tt*tp*exp(-bigDelta*tt)*exp(-11*tt*tr)*exp(-12*tt*tp) + 12*tt*tp*exp((11*tr+12*tp-bigDelta)*tt) + 60)./(3*tt.^4*tp.^2)) ; 
        tmp = tmp + Bk*1/2*(24.*exp(-(tt).*tp) - 24.*exp(-(tt).*tr) - 72.*exp(-2.*(tt).*tr) - 96.*exp(-bigDelta.*(tt)) - 168.*(tt).*tr + 12.*exp(-(tt).*(bigDelta-tr)) + 12.*exp(-(tt).*(bigDelta+tr)) + 36.*exp(-(tt).*(bigDelta+2.*tr)) + 36.*exp(-(tt).*(bigDelta-2.*tr)) - 12.*exp(-(tt).*(bigDelta-tp)) - 12.*exp(-(tt).*(bigDelta+tp)) - 24.*exp(-(tt).*(tr+tp)) + 60.*exp(-(tt).*(tr+2.*tp)) - 24.*exp(-(tt).*(2.*tr+tp)) + 24.*exp(-(tt).*(3.*tr+tp)) - 120.*exp(-(tt).*(3.*tr+2.*tp)) - 24.*exp(-(tt).*(3.*tr+3.*tp)) + 24.*exp(-(tt).*(4.*tr+3.*tp)) + 60.*exp(-(tt).*(5.*tr+2.*tp)) - 48.*exp(-(tt).*(4.*tr+4.*tp)) + 24.*exp(-(tt).*(5.*tr+3.*tp)) - 24.*exp(-(tt).*(6.*tr+3.*tp)) + 96.*exp(-(tt).*(6.*tr+4.*tp)) + 24.*exp(-(tt).*(6.*tr+5.*tp)) - 24.*exp(-(tt).*(7.*tr+5.*tp)) - 48.*exp(-(tt).*(8.*tr+4.*tp)) + 36.*exp(-(tt).*(7.*tr+6.*tp)) - 24.*exp(-(tt).*(8.*tr+5.*tp)) + 24.*exp(-(tt).*(9.*tr+5.*tp)) - 72.*exp(-(tt).*(9.*tr+6.*tp)) - 24.*exp(-(tt).*(9.*tr+7.*tp)) + 24.*exp(-(tt).*(10.*tr+7.*tp)) + 36.*exp(-(tt).*(11.*tr+6.*tp)) - 24.*exp(-(tt).*(10.*tr+8.*tp)) + 24.*exp(-(tt).*(11.*tr+7.*tp)) - 24.*exp(-(tt).*(12.*tr+7.*tp)) + 48.*exp(-(tt).*(12.*tr+8.*tp)) + 24.*exp(-(tt).*(12.*tr+9.*tp)) - 24.*exp(-(tt).*(13.*tr+9.*tp)) - 24.*exp(-(tt).*(14.*tr+8.*tp)) + 12.*exp(-(tt).*(13.*tr+10.*tp)) - 24.*exp(-(tt).*(14.*tr+9.*tp)) + 24.*exp(-(tt).*(15.*tr+9.*tp)) - 24.*exp(-(tt).*(15.*tr+10.*tp)) - 24.*exp(-(tt).*(15.*tr+11.*tp)) + 24.*exp(-(tt).*(16.*tr+11.*tp)) + 12.*exp(-(tt).*(17.*tr+10.*tp)) + 24.*exp(-(tt).*(17.*tr+11.*tp)) + 12.*exp(-(tt).*(17.*tr+12.*tp)) - 24.*exp(-(tt).*(18.*tr+11.*tp)) - 24.*exp(-(tt).*(18.*tr+12.*tp)) + 12.*exp(-(tt).*(19.*tr+12.*tp)) + 116.*(tt).^3.*tr.^3 + 144.*(tt).^3.*tr.^2.*tp + 12.*exp(-(tt).*(bigDelta-tr-tp)) - 30.*exp(-(tt).*(bigDelta-tr-2.*tp)) + 12.*exp(-(tt).*(bigDelta+tr+tp)) + 12.*exp(-(tt).*(bigDelta-2.*tr-tp)) - 30.*exp(-(tt).*(bigDelta+tr+2.*tp)) + 12.*exp(-(tt).*(bigDelta+2.*tr+tp)) - 12.*exp(-(tt).*(bigDelta-3.*tr-tp)) - 12.*exp(-(tt).*(bigDelta+3.*tr+tp)) + 60.*exp(-(tt).*(bigDelta+3.*tr+2.*tp)) + 60.*exp(-(tt).*(bigDelta-3.*tr-2.*tp)) + 12.*exp(-(tt).*(bigDelta+3.*tr+3.*tp)) + 12.*exp(-(tt).*(bigDelta-3.*tr-3.*tp)) - 12.*exp(-(tt).*(bigDelta+4.*tr+3.*tp)) - 12.*exp(-(tt).*(bigDelta-4.*tr-3.*tp)) - 30.*exp(-(tt).*(bigDelta+5.*tr+2.*tp)) - 30.*exp(-(tt).*(bigDelta-5.*tr-2.*tp)) + 24.*exp(-(tt).*(bigDelta+4.*tr+4.*tp)) + 24.*exp(-(tt).*(bigDelta-4.*tr-4.*tp)) - 12.*exp(-(tt).*(bigDelta+5.*tr+3.*tp)) - 12.*exp(-(tt).*(bigDelta-5.*tr-3.*tp)) + 12.*exp(-(tt).*(bigDelta+6.*tr+3.*tp)) + 12.*exp(-(tt).*(bigDelta-6.*tr-3.*tp)) - 48.*exp(-(tt).*(bigDelta+6.*tr+4.*tp)) - 48.*exp(-(tt).*(bigDelta-6.*tr-4.*tp)) - 12.*exp(-(tt).*(bigDelta+6.*tr+5.*tp)) - 12.*exp(-(tt).*(bigDelta-6.*tr-5.*tp)) + 12.*exp(-(tt).*(bigDelta+7.*tr+5.*tp)) + 12.*exp(-(tt).*(bigDelta-7.*tr-5.*tp)) + 24.*exp(-(tt).*(bigDelta+8.*tr+4.*tp)) + 24.*exp(-(tt).*(bigDelta-8.*tr-4.*tp)) - 18.*exp(-(tt).*(bigDelta+7.*tr+6.*tp)) - 18.*exp(-(tt).*(bigDelta-7.*tr-6.*tp)) + 12.*exp(-(tt).*(bigDelta+8.*tr+5.*tp)) + 12.*exp(-(tt).*(bigDelta-8.*tr-5.*tp)) - 12.*exp(-(tt).*(bigDelta+9.*tr+5.*tp)) - 12.*exp(-(tt).*(bigDelta-9.*tr-5.*tp)) + 36.*exp(-(tt).*(bigDelta+9.*tr+6.*tp)) + 36.*exp(-(tt).*(bigDelta-9.*tr-6.*tp)) + 12.*exp(-(tt).*(bigDelta+9.*tr+7.*tp)) + 12.*exp(-(tt).*(bigDelta-9.*tr-7.*tp)) - 12.*exp(-(tt).*(bigDelta+10.*tr+7.*tp)) - 12.*exp(-(tt).*(bigDelta-10.*tr-7.*tp)) - 18.*exp(-(tt).*(bigDelta+11.*tr+6.*tp)) - 18.*exp(-(tt).*(bigDelta-11.*tr-6.*tp)) + 12.*exp(-(tt).*(bigDelta+10.*tr+8.*tp)) + 12.*exp(-(tt).*(bigDelta-10.*tr-8.*tp)) - 12.*exp(-(tt).*(bigDelta+11.*tr+7.*tp)) - 12.*exp(-(tt).*(bigDelta-11.*tr-7.*tp)) + 12.*exp(-(tt).*(bigDelta+12.*tr+7.*tp)) + 12.*exp(-(tt).*(bigDelta-12.*tr-7.*tp)) - 24.*exp(-(tt).*(bigDelta+12.*tr+8.*tp)) - 24.*exp(-(tt).*(bigDelta-12.*tr-8.*tp)) - 12.*exp(-(tt).*(bigDelta+12.*tr+9.*tp)) - 12.*exp(-(tt).*(bigDelta-12.*tr-9.*tp)) + 12.*exp(-(tt).*(bigDelta+13.*tr+9.*tp)) + 12.*exp(-(tt).*(bigDelta-13.*tr-9.*tp)) + 12.*exp(-(tt).*(bigDelta+14.*tr+8.*tp)) + 12.*exp(-(tt).*(bigDelta-14.*tr-8.*tp)) - 6.*exp(-(tt).*(bigDelta+13.*tr+10.*tp)) - 6.*exp(-(tt).*(bigDelta-13.*tr-10.*tp)) + 12.*exp(-(tt).*(bigDelta+14.*tr+9.*tp)) + 12.*exp(-(tt).*(bigDelta-14.*tr-9.*tp)) - 12.*exp(-(tt).*(bigDelta+15.*tr+9.*tp)) - 12.*exp(-(tt).*(bigDelta-15.*tr-9.*tp)) + 12.*exp(-(tt).*(bigDelta+15.*tr+10.*tp)) + 12.*exp(-(tt).*(bigDelta-15.*tr-10.*tp)) + 12.*exp(-(tt).*(bigDelta+15.*tr+11.*tp)) + 12.*exp(-(tt).*(bigDelta-15.*tr-11.*tp)) - 12.*exp(-(tt).*(bigDelta+16.*tr+11.*tp)) - 12.*exp(-(tt).*(bigDelta-16.*tr-11.*tp)) - 6.*exp(-(tt).*(bigDelta+17.*tr+10.*tp)) - 6.*exp(-(tt).*(bigDelta-17.*tr-10.*tp)) - 12.*exp(-(tt).*(bigDelta+17.*tr+11.*tp)) - 12.*exp(-(tt).*(bigDelta-17.*tr-11.*tp)) - 6.*exp(-(tt).*(bigDelta+17.*tr+12.*tp)) - 6.*exp(-(tt).*(bigDelta-17.*tr-12.*tp)) + 12.*exp(-(tt).*(bigDelta+18.*tr+11.*tp)) + 12.*exp(-(tt).*(bigDelta-18.*tr-11.*tp)) + 12.*exp(-(tt).*(bigDelta+18.*tr+12.*tp)) + 12.*exp(-(tt).*(bigDelta-18.*tr-12.*tp)) - 6.*exp(-(tt).*(bigDelta+19.*tr+12.*tp)) - 6.*exp(-(tt).*(bigDelta-19.*tr-12.*tp)) + 96)./(3.*(tt).^4.*tr.^2) ; 

end

S = exp(-gamma^2*Grad^2*tmp);

function [Bk, lambda_k] = calcBk(diameter, k)

%root_cyl = HL_FindRealRoots(@(x) x*(besselj(1/2,x)-besselj(5/2,x))-besselj(3/2,x),0.1,100,200);
root_cyl = [ 2.082,  5.941, 9.206, 12.405, 15.580, 18.743, 21.900, 25.053, 28.204, 31.353, 34.500, 37.646, 40.792, 43.937,...
    47.082, 50.226, 53.370, 56.514, 59.657, 62.801, 65.944, 69.087, 72.229, 75.372, 78.515, 81.657, 84.800, 87.942, 91.085  ]';
%root_cyl = FindRealRoots(@(x) (besselj(2,x)-besselj(0,x)),0.1,100,200);

%Bk = 2*(diameter/2/root_cyl(k)).^2./(root_cyl(k)^2 - 1); %cylinder
%lambda_k = (root_cyl(k)/(diameter/2)).^2; % cylinder
Bk = 2*(diameter/2/root_cyl(k)).^2./(root_cyl(k)^2 - 2); %cell
lambda_k = (root_cyl(k)/(diameter/2)).^2; % cell
%lambda_k = pi^2*(2*k-1)^2/diameter^2;
%a = 0e-6; % from um to m
%b = 20e-6; % from um to m
%Bk = 2*a^3*b^3*((besselj(0,lambda_k*a)-besselj(2,lambda_k*a))/2-(besselj(0,lambda_k*b)-besselj(2,lambda_k*b))/2)^2/...
%   (lambda_k^2*(b^3-a^3)*((a^3*(lambda_k^2*b^2-2)*(besselj(0,lambda_k*a)-besselj(2,lambda_k*a))/2)-...
%   b^3*(lambda_k^2*a^2-2)*(besselj(0,lambda_k*b)-besselj(2,lambda_k*b))/2));
