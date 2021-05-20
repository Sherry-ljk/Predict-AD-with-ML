clear
clc
if ispc
else
    parpool('local',20)
end

iter_max = 200;
n = 1000;
repeat_num = 20;

lb = [10^(-7) 0.001 10 0.001 0.01 0.0 0.1]; % lower bounds
ub = [10^(-4)  2   100  2     1   0.2 1.0 ]; % upper bounds
rng((now*1000-floor(now*1000))*100000)
tic
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
filename = ['opt_Lorenz_15_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

func = @(x) (func_Lorenz_train_repeat(x,n,repeat_num));
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(func,lb,ub,options);
toc

save(filename)
if ~ispc
    exit;
end
