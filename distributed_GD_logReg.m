function [obj_GD, loss_GD, Iter] = distributed_GD_logReg(XX,YY, no_workers, num_feature, noSamples, num_iter, obj0...
    , acc, lr, lambda_logistic)

Iter= num_iter;              
out_central=zeros(num_feature,1);
grad=zeros(num_feature,no_workers);

max_iter = num_iter;



for i = 1:max_iter

         
     for ii =1:no_workers

         
         first = (ii-1)*noSamples+1;
         last = first+noSamples-1;
        
        Z=XX(first:last,1:num_feature);
        Y=YY(first:last);


        grad(:,ii)=-1/noSamples*(Z'*(Y./(1+exp(Y.*(Z*out_central)))))+lambda_logistic*out_central;
        %grad(:,ii)=1/noSamples*(Z'*Z*out_central-Z'*Y);%
  
        
        
        
     end
     


     
    out_central = out_central- lr/no_workers* sum(grad,2);    
    
    
         
        % final_obj = 0;
        % for ii =1:no_workers
        %     first = (ii-1)*noSamples+1;
        %     last = first+noSamples-1;
        %     %final_obj = final_obj + 0.5*norm(XX(first:last,1:num_feature)*out_central - YY(first:last))^2;
        %     final_obj =final_obj+1/no_workers*(lambda_logistic*0.5*norm(out_central)^2+...
        %         1/noSamples*sum(log(1+exp(-YY(first:last).*(XX(first:last,1:num_feature)*out_central)))));
        % 
        % end

        final_obj=lambda_logistic*0.5*norm(out_central)^2+1/(noSamples*no_workers)*sum(log(1+exp(-YY.*(XX*out_central))));



        obj_GD(i)=final_obj;
        loss_GD(i)=abs(final_obj-obj0);
        % fprintf('The iteration number is: %d\n', i);
        %fprintf('The loss value is: %f\n', labs(final_obj-obj0));
        % fprintf('The objective value is: %f\n', obj_GD(i));
        %printLoss=loss_GD(i)
        if(loss_GD(i) <= acc)
            Iter = i;
            break;
        end
        
end   
    

end
     




