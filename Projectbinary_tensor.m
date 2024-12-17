function [res,res_cluster,iter,obj2,Phi,Bi] = Projectbinary_tensor(Phi,label,alpha,l,num_anchor)
%% initialize
maxIter = 50 ; % the number of iterations
m = num_anchor;
num_class = length(unique(label));
num_view = length(Phi);
num_sample = size(label,1);
  
%% ---------Initial C, G, B with its parameters--------------
innerMax = 10;
N = size(Phi{1},2);
for v = 1:num_view
    rand('seed',100);
    sel_sample = Phi{v}(:,randsample(N, num_anchor),:); 
    [pcaW, ~] = eigs(cov(sel_sample'), l);   
    B{v} = sign(pcaW'*Phi{v});  
end

rand('seed',500);
C = B{1}(:,randsample(N, num_class));
HamDist = 0.5*(l - B{1}'*C);
[~,ind] = min(HamDist,[],2);
G = sparse(ind,1:N,1,num_class,N,N);
G = full(G);

%% initialize Q,A,Y,E,J
for v = 1:num_view
   Q{v} = zeros(l,m); 
   A{v} = zeros(l,m);
   Y{v} = zeros(l,m);
   %initialize
   E{v} = zeros(l,num_sample);
   J{v}=zeros(l,num_sample);
end


opt.disp = 0;

mu = 10e-4; max_mu = 10e10; pho_mu = 2;
flag = 1;
iter = 0;
res=[];
tic;
while flag
    iter = iter + 1;
    %---------Update1 Qv--------------
    for v=1:num_view
        temp_Q1{v} = 2*alpha*B{v}*Phi{v}'+mu*(A{v}-Y{v}/mu);
        temp_Q2{v} = 2*alpha*Phi{v}*Phi{v}'+mu*eye(m);
        Q{v} = temp_Q1{v}*pinv(temp_Q2{v}) ;
    end

    %---------Update4 Bv--------------
    for v = 1:num_view
       temp_B =  Q{v}*Phi{v}+E{v}-J{v}/mu;
       B{v} = sign(temp_B); B{v}(B{v}==0) = -1;
    end
    %---------Update2 A_tensor--------------
         for v = 1:num_view
             temp_tensor_Q{v} = Q{v}';
             temp_tensor_Y{v} = Y{v}';
         end
    Q_tensor = cat(3, temp_tensor_Q{:,:});
    Y_tensor = cat(3, temp_tensor_Y{:,:});
    Ten_A = Q_tensor+Y_tensor/mu;
    Ten_A = shiftdim(Ten_A, 1);
    [A_tensor,~,~] = prox_n_itnn(Ten_A,1/mu);
    A_tensor = shiftdim(A_tensor, 2);

   %---------Update3 C_tensor --------------
         for v = 1:num_view
             B{v} = B{v}';
             J{v} = J{v}';
         end
    B_tensor = cat(3, B{:,:});
    J_tensor = cat(3, J{:,:});
    Ten_C = B_tensor+J_tensor/mu;
    Ten_C = shiftdim(Ten_C, 1);
    [E_tensor,~,~] = prox_n_itnn(Ten_C,1/mu);
    E_tensor = shiftdim(E_tensor, 2);

    %% solve  Y_tensor and  penalty parameters        
    Y_tensor = Y_tensor + mu*(Q_tensor - A_tensor);
    J_tensor = J_tensor + mu*(B_tensor - E_tensor);
    mu = min(mu*pho_mu, max_mu);
    
    for v = 1:num_view
        Q{v} = Q_tensor(:,:,v)';
        A{v} = A_tensor(:,:,v)';
        Y{v} = Y_tensor(:,:,v)';
        B{v} = B_tensor(:,:,v)';
        E{v} = E_tensor(:,:,v)';
        J{v} = J_tensor(:,:,v)';
    end

    term1 = 0;
    for v = 1:num_view
        term1 = term1 + norm(Q{v}*Phi{v}-B{v},'fro')^2;
    end
    obj(iter) = term1;
       
    if (iter>1)
            obj2(iter) = abs((obj(iter-1)-obj(iter))/(obj(iter-1)));
    end
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-5 || iter>maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end
disp('----------Main Iteration Completed and clustering----------');
    %---------Update C and G--------------
    B_temp = 0;
    for v = 1:num_view
        B_temp = B_temp + B{v};
    end
    B_average = B_temp/num_view;
    for iterInner = 1:innerMax
        % For simplicity, directly using DPLM here
        C = sign(B_average*G'); C(C==0) = 1;
        rho = .001; mu2 = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B_average*G' + rho*repmat(sum(C),l,1);
            C    = sign(C-1/mu2*grad); C(C==0) = 1;
        end
        HamDist = 0.5*(l - B_average'*C); % Hamming distance referring to "Supervised Hashing with Kernels"
        [~,indx] = min(HamDist,[],2);
        G = sparse(indx,1:N,1,num_class,N,N);
    end

Bi = B_average;
[~,pred_label] = max(G,[],1);
time = toc;
fprintf('time = %.4f\n',time);
res_cluster = Clustering8Measure(label, pred_label)%[ACC nmi Purity Fscore Precision AR]

         
         
    
