clear all;close all;
load('dna.mat')
[n,m]=size(Z);
max_num_of_data=2000;
if n>max_num_of_data
    selected=randperm(n,max_num_of_data);
    Z=Z(selected,:);
    y=y(selected,:);
    n=max_num_of_data;
end
if min(y)<1
    y=y+1;
end
k=max(y);
T1 = 200;      % iters of row update
T4 = 100;       % iters of BFGS
T5 = 100;       % iters of LBFGS
T6 = 200;       % (*rate6=) iters of saga
rate6 = 20;     % time per rate6 iters in saga
lambda = 0;

for ss = 1:1
    randn('seed',ss);rand('seed',ss)
    
    X = Z;%rand(n,m); % samples
    
    w_init = randn(k,m);
    w_init(k,:)=0;
    
    obj1 = zeros(T1,1);
    time1 = zeros(T1,1);
    obj11 = zeros(T1,1);
    time11 = zeros(T1,1);
    obj4 = zeros(T4,1);
    time4 = zeros(T4,1);
    
    for i=1:k
        y_b(:,i)=(y==i);
    end
    L_all = sum(X.^2,"all")./4;
    L = sum(X.^2)./4;
    L2=norm(X,2)^2/4;
    Ln = max(sum(X.^2,2)/4);
    w = w_init;
    tic
    for i=1:T1*rate6
        % t_total = sum(abs(w),'all');
        if mod(i,rate6)==1
            obj1(floor(i/rate6)+1) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
            time1(floor(i/rate6)+1) = toc;
        end
        %tmp = randi(n);
        for h=1:m
            dw = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X(:,h);
            % P = t_total-abs(w(h,:));
            t = w(1:k-1,h) - dw./L(h);
            % t_abs = abs(t);
            w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
        end
    end
    
    w = w_init;
    tic
    for i=1:T1*rate6
        % t_total = sum(abs(w),'all');
        if mod(i,rate6)==1
            obj11(floor(i/rate6)+1) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
            time11(floor(i/rate6)+1) = toc;
        end
        h = randi(m);
        %for h=1:m
            dw = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X(:,h);
            % P = t_total-abs(w(h,:));
            t = w(1:k-1,h) - dw./L(h);
            % t_abs = abs(t);
            w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
        %end
    end
    opts = struct();
    opts.xtol = 0;
    opts.gtol = 0;
    opts.ftol = 0;
    opts.maxit = T5;
    opts.record  = 0;
    opts.m = 20;
    
    fun = @(w)obj(w,X,y,y_b,lambda);
    hess = @(w,u)hessian(w,X,u);
    x0 = w_init(1:k-1,:);
    x0 =x0(:);
    [x1,~,~,out1] = fminLBFGS_Loop(x0,fun,opts);
    fprintf("lbfgs");
    opts = struct();
    opts.xtol = 1e-8;
    opts.gtol = 1e-6;
    opts.ftol = 1e-16;
    opts.maxit = T5;
    opts.verbose = 0;
    [x2,out2] = fminNewton(x0,fun,hess,opts);
    fprintf("newton")
    opts = struct();
    opts.xtol = 1e-8;
    opts.gtol = 1e-6;
    opts.ftol = 1e-16;
    opts.maxit = T5;
    opts.record  = 0;
    opts.verbose = 0;
    opts.Delta = sqrt(n);
    c1 = 1e-4;
    eta = 1.5;
    c2 = .9;
    [x3,out3] = fminTR(x0,fun,hess,opts);
    fprintf("tr")
           
    figure;
    semilogy(-obj1,"LineWidth",1.5);
    hold on;
    semilogy(-obj11,"LineWidth",1.5);
    semilogy(out1.f,"LineWidth",1.5);
    semilogy(out2.f,"LineWidth",1.5);
    semilogy(out3.f,"LineWidth",1.5);
    hold off;
    grid on;
    legend("Alg1","Alg2","LBFGS","NewtonConjugate","TrustRegion");
    title("Iteration")
    
    figure;
    semilogy(time1,-obj1,"LineWidth",1.5);
    hold on;
    semilogy(time11,-obj11,"LineWidth",1.5);
    semilogy(out1.time,out1.f,"LineWidth",1.5);
    semilogy(out3.time,out3.f,"LineWidth",1.5);
    hold off;
    grid on;
    legend("Alg1","Alg2","LBFGS","TrustRegion");
    title("Time")
end
function [f,g] = obj(w,X,y,y_b,lambda)
m = size(X,2);
w_m = [reshape(w,[],m);zeros(1,m)];
k = size(w_m,1);
f = -trace(X*w_m(y,:)')+sum(log(1+sum(exp(w_m*X'))))+lambda*sum(abs(w),'all');
g = (exp(w_m(1:k-1,:)*X')./(1+sum(exp(w_m(1:k-1,:)*X')))-y_b(:,1:k-1)')*X;
g = g(:)+lambda*sign(w);
end

function result = hessian(w,X,u)
m = size(X,2);
w_m = [reshape(w,[],m)];
k = size(w_m,1);
exp_sum = (1+sum(exp(w_m*X')));
h = zeros(m*k);
for i = 1:k
    for j = 1:k
        if i==j
            tmp = (exp(w_m(i,:)*X')./exp_sum)';
            h((i-1)*m+1:i*m,(i-1)*m+1:i*m)=X'*(tmp.*(1-tmp).*X);
        else
            h((i-1)*m+1:i*m,(j-1)*m+1:j*m) = X'*((exp(w_m(i,:)*X').*exp(w_m(j,:)*X')./(exp_sum.^2))'.*X);
        end
    end
end
result = h*u;
end