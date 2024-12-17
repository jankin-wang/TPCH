function [A] = constructA(ind)
[n,m] = size(ind);
for i=1:m
    existsamples = find(ind(:,i)==1);
    number = length(existsamples);
    Ai=zeros(1,n);
    for j=1:number
        Ai(1,existsamples(j)) =1;
    end
    A{i} = Ai;
end

