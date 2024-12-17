   function [A] = Construct_indicate_matrix(m,numclass,numsample)

   
        each_clusternum  = numsample/m*(m/numclass);
            b =1;
            c = numclass;
            d=1;
        for j = 1:each_clusternum:numsample
            colunm1 = each_clusternum*(c-numclass)+1;
            colunm2 = each_clusternum*d;
            low1 = (b-1)*(m/numclass)+1;
            low2 = b*(m/numclass);
           A(low1:low2,colunm1:colunm2) = 1;
                c=c+1;
                b=b+1;
                d=d+1;
        end