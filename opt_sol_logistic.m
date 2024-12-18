function out=opt_sol_logistic(XX,YY, dim, lambda, num_workers, num_sample)
s = dim;

cvx_begin %quiet
%cvx_precision %low%high%low
cvx_solver SeDuMi%mosek%SDPT3%mosek%SeDuMi%mosek%SDPT3
variable x(s)

    obj=lambda*0.5*sum_square(x)+1/(num_sample*num_workers)*sum(log(1+exp(-YY.*(XX*x))));

minimize(obj)
cvx_end

out =lambda*0.5*sum_square(x)+1/(num_sample*num_workers)*sum(log(1+exp(-YY.*(XX*x))));



end

