function [obj_GD, loss_GD, Iter] = distributed_GD(XX,YY, no_workers, num_feature, noSamples, num_iter, obj0...
    , acc, lr)

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
        grad(:,ii)=1/noSamples*(Z'*Z*out_central-Z'*Y);%
  
        
        
        
     end
     


     
    out_central = out_central- lr/no_workers* sum(grad,2);    
    
    
         
        final_obj = 0;
        for ii =1:no_workers
            first = (ii-1)*noSamples+1;
            last = first+noSamples-1;
            final_obj = final_obj + 1/(no_workers*noSamples)*0.5*norm(XX(first:last,1:num_feature)*out_central - YY(first:last))^2;
        end
        obj_GD(i)=final_obj;
        loss_GD(i)=abs(final_obj-obj0);            
        if(loss_GD(i) <= acc)
            Iter = i;
            break;
        end
        
end   
    

end
     




