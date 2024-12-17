clear;
clc;

addpath(genpath('./'));

resultdir3 = 'res_all_parameters/';
if (~exist('res_all_parameters', 'file'))
    mkdir('res_all_parameters');
    addpath(genpath('res_all_parameters/'));
end

datadir='Datasets/'; 
Dataname = '100leaves';
%%  Run
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
        datafile = [datadir, str2mat(Dataname),'.mat'];
        load(datafile);
        filename = [str2mat(Dataname),'.mat'];
        %% -----------------data preparation...-----------------
        cls_num = length(unique(Y));         
        k= cls_num;
        sample_num = length(Y);
        numview = length(X);
        tic;
        if(size(Y,2)~=1)                    
              Y = Y';
        end
        if ~isempty(find(Y==0,1))          
                Y = Y + 1;
        end
        for v = 1:numview
         if size(X{v},1)~=sample_num
               X{v} = X{v}';
         end
         X{v} = NormalizeFea(X{v},0);
        end
        
        res_all = [];
        time1 = toc;
        maxAcc = 0;

        anchor_rate = [0.2];

      %% -----------------Anchor selection preparation-----------------%%
        XX = [];
        for v = 1:numview
            XX = [XX X{v}];
        end
        num_anchor = fix(sample_num*anchor_rate);
        [~,ind,~] = VDA(XX,num_anchor); 
        for v = 1:numview
            Anchor{v} = X{v}(ind, :);
        end
        
        fprintf('Nonlinear Anchor Embedding...\n');
    for it = 1:numview
        dist = EuDist2(X{it},Anchor{it},0);
        sigma = mean(min(dist,[],2).^0.5)*2;
        feaVec = exp(-dist/(2*sigma*sigma));
        X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));
    end
    clear feaVec dist sigma dist Anchor it

        savedata = [];
        idx = 1;
                alpha = 0.1;
                l = 64;

     disp([char(Dataname),'-l2=', num2str(alpha), '-l3=', num2str(l), '-l4=', num2str(num_anchor)]);
                tic;
                para.c = cls_num; 
                para.k = num_anchor; 
           [res,res_cluster,iter,obj,X_complete,Bi] = Projectbinary_tensor(X,Y,alpha,l,num_anchor); 
                time = toc;
                runtime(idx) = time;
                savedata = [savedata;num_anchor, alpha, l, num_anchor,time,res_cluster];

                disp(['runtime:', num2str(runtime(idx))])    

        tm=datestr(now,'yyyy-mm-dd_HH_MM_SS');    
        %% save all parameters
        res_file = fullfile('./res_all_parameters/', [tm, char(Dataname), num2str(num_anchor), 'itnn', '.mat']);%res file needed
        save(res_file, 'savedata');


