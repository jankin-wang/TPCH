function[X] = update_Xm(X,missing_ind,miss_index,alpha,temp_PZ,W,P,Z)

numview = length(X);

%temp_PZ weighted
for v = 1:numview
      temp_Xm = 0;
      for i = 1:numview
        temp_Xm = temp_Xm + alpha(i)^2 * temp_PZ{i}(:,miss_index{v});
      end
       X{v}(:,miss_index{v}) = W{v} * temp_Xm;
end

I=1;






