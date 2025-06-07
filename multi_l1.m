clear variables;
n = 100;   % # of samples
m = 50;    % # of features
k = 10;     % # of classes
X = rand(n,m); % samples
T1 = 20;      % iters of row update
T2 = 70;       % iters of matrix update 
y = randi(k,[n,1]); % labels
w_init = randn(k,m);
w_init(k,:)=0;
lambda = 1;

obj1 = zeros(T1,1);
time1 = zeros(T1,1);
obj2 = zeros(T2,1);
time2 = zeros(T2,1);

i=1:k;
y_b(:,i)=(y==i);

L_all = sum(X.^2,"all")./4;
L = sum(X.^2)./4;
L2=norm(X,2)^2/4;
w = w_init;
tic
dw=zeros(k-1,m);
for i=1:T1
    % t_total = sum(abs(w),'all');
    obj1(i) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
    time1(i) = toc;
    for h=1:m
        dw = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X;
        % P = t_total-abs(w(h,:));
        t = w(1:k-1,h) - dw(:,h)./L(h);
        % t_abs = abs(t);
        w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
    end
end

w = w_init;
tic
for i=1:T2
    obj2(i) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
    time2(i) = toc;
    dw = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X;
    t = w(1:k-1,:) - dw./L2;
    w(1:k-1,:) = sign(t).*max(abs(t)-lambda./L2,0);
end

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

counter = sum(obj1(1:end-1)>obj1(2:end));

figure;
semilogy(obj1,"LineWidth",1.5);
hold on;
semilogy(obj2,"LineWidth",1.5);
semilogy(-out.f,"LineWidth",1.5);
hold off;
legend("Vector Update","Matrix","BFGS");
title("Iteration")

figure;
semilogy(time1,obj1,"LineWidth",1.5);
hold on;
semilogy(time2,obj2,"LineWidth",1.5);
semilogy(out.time,-out.f,"LineWidth",1.5);
hold off;
legend("Vector Update","Matrix","BFGS");
title("Time")

function [f,g] = obj(w,X,y,y_b,lambda)
    m = size(X,2);
    w_m = reshape(w,[],m);
    f = -trace(X*w_m(y,:)')+sum(log(1+sum(exp(w_m*X'))))+lambda*sum(abs(w),'all');
    g = (exp(w_m*X')./(1+sum(exp(w_m*X')))-y_b')*X;
    g = g(:);
end