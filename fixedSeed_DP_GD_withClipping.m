function [obj_GD, loss_GD, Iter] = fixedSeed_DP_GD_withClipping(XX,YY, no_workers, num_feature, noSamples, num_iter, obj0...
    , acc, lr,epsilon, delta, c)

Iter= num_iter;              
global_model=zeros(num_feature,1);
grad=zeros(num_feature,no_workers);

max_iter = num_iter;
%c=100;

for i = 1:max_iter

     %rng(randi([1 100]));
     sigma = 2*c*sqrt(2*log(1.25/delta))/(epsilon*noSamples);
     % %sigma = c*sqrt(2*log(1.25/delta))/(epsilon);
     % %noise = wgn(num_feature,1,power);
     noise = normrnd(0,sigma,[num_feature,1]);

    % sensitivity = 2*c/noSamples;
    % sigma = sqrt(2)*sensitivity/epsilon;
    % noise = sigma.*randl(num_feature,1);



     for ii =1:no_workers

         
         first = (ii-1)*noSamples+1;
         last = first+noSamples-1;
        
        Z=XX(first:last,1:num_feature);
        Y=YY(first:last);
        

        sum_grad=Z'*Z*global_model-Z'*Y;
        norm_sumgrad=norm(sum_grad);
        if norm_sumgrad > c
                clipped_sumgrad = c * sum_grad/norm_sumgrad;

             else
                clipped_sumgrad = sum_grad;
         end
        grad(:,ii)=clipped_sumgrad*1/noSamples;
        %grad(:,ii)=1/noSamples*(Z'*Z*global_model-Z'*Y);%
        grad(:,ii)=grad(:,ii)+noise;
        
        
        
     end
     


     
    global_model = global_model- lr/no_workers* sum(grad,2); % at the PS
    global_model = global_model+lr* noise; % at each client
    
    
         
        final_obj = 0;
        for ii =1:no_workers
            first = (ii-1)*noSamples+1;
            last = first+noSamples-1;
            final_obj = final_obj + 1/(no_workers*noSamples)*0.5*norm(XX(first:last,1:num_feature)*global_model - YY(first:last))^2;
        end
        obj_GD(i)=final_obj;
        loss_GD(i)=abs(final_obj-obj0);            
        if(loss_GD(i) <= acc)
            Iter = i;
            break;
        end
        
end   
    

end
     




