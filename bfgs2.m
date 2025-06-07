
clear variables;

n = 100;   % # of samples
m = 50;    % # of features
k = 10;     % # of classes
X = rand(n,m); % samples
T1 = 200;      % iters 
y = randi(k,[n,1]); % labels
w_init = randn(k,m);
w_init(k,:)=0;
lambda = 1;

opts = struct();
opts.xtol = 1e-6;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.maxit = 2000;
opts.record  = 0;
opts.m = 5;

y2 = y(y~=k);
X2 = X(y~=k,:);
i=1:k-1;
y_b2(:,i)=(y2==i);
fun = @(w)obj(w,X2,y2,y_b2,lambda);
x0 = w_init(1:k-1,:);
x0 =x0(:);
[tmp,gtmp] = fun(x0);
[x,~,~,out] = fminLBFGS_Loop(x0,fun,opts);

function [f,g] = obj(w,X,y,y_b,lambda)
    m = size(X,2);
    w_m = reshape(w,[],m);
    f = -trace(X*w_m(y,:)')+sum(log(1+sum(exp(w_m*X'))))+lambda*sum(abs(w),'all');
    g = (exp(w_m*X')./(1+sum(exp(w_m*X')))-y_b')*X;
    g = g(:);
end

plot(out.time,out.f)