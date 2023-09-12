function model = team_training_code(input_directory,output_directory, verbose) % train_EEG_classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Train EEG classifiers and obtain the models
% Inputs:
% 1. input_directory
% 2. output_directory
%
% Outputs:
% 1. model: trained model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose>=1
    disp('Finding challenge data...')
end

% Find the folders
patient_ids=dir(input_directory);
patient_ids=patient_ids([patient_ids.isdir]==1);
patient_ids(1:2)=[]; % Remove "./" and "../" paths
patient_ids={patient_ids.name};
num_patients = length(patient_ids);

% Create a folder for the model if it doesn't exist
if ~isfolder(output_directory)
    mkdir(output_directory)
end
fprintf('Loading data for %d patients...\n', num_patients)

channels = {'Fp1', 'Fp2', 'F3','F4'};
test_chs_L=length(channels);
[PartA, PartB]=z_bipartition_TE(test_chs_L);

j_cnt=0;
for j=1:num_patients

    if verbose>1
        fprintf('%d/%d \n',j,num_patients)
    end

    % Extract features
    patient_id=patient_ids{j};

    current_features=get_features(input_directory,patient_id,PartA,PartB,channels);

    if size(current_features,1)>0
        if sum(sum(isnan(current_features')))~=0
            current_features(sum(isnan(current_features'))>0,:)=[];
        end
        if size(current_features,1)>0
            c_d_num=[];
            for c_i=1:size(current_features,2)-1
                d_tmp=find(abs(current_features(:,c_i)) > 999);
                c_d_num=vertcat(c_d_num,d_tmp);
                clear d_tmp
            end
            if sum(c_d_num)~=0
                current_features(c_d_num,:)=[];
            end
            if size(current_features,1)>0

                if size(current_features,1)>0
                    j_cnt=j_cnt+1;

                    [patient_metadata,~]=load_challenge_data(input_directory,patient_id);
                    current_outcome=get_outcome(patient_metadata);

                    current_features(:,size(current_features,2)+1)=current_outcome;
                    features_struct{j_cnt}=current_features;

                end
            end
        end
    end

    clear current_features patient_id hos_Num
    clear patient_metadata current_outcome c_d_num
end


median_bowl=[0.0135834774755276	0.0534925919728622	0.118902167100248	36.9485539831741	39.8491382433925	20.4846472399850	23.6807960176608	8.11328125000000	8.25781250000000	8.91796875000000	9.10644531250000	9.70261436570684	9.70324585022036	9.29379984973723	9.32579317624609	0.937136158710965	2.18978102189781	2.70270270270270	3.50877192982456	0	1.26115180025903	1.32563204137101	0.980392156862745	0	1.61380235172861	1.65266917916043	1.65178901640987	1.54433412633408	1.44306693780431	1.47260015294299	1.45285761779823	1.29919958337640	0.0898696159686560	0.0908590021983754	0.0858913655268553	0.0744926384807600	0.0543367764720562	0.0545033744349376	0.0524064794530345	0.0459832914457559];



%% train model
e_div_L=4;
T_D_L=3;
channel_L=4;


features_all_72=[];
features_all_48=[];
features_all_24=[];


features_all_24_sub=[];
features_all_48_sub=[];
features_all_72_sub=[];


t_f_size=20;
for f_j=1:length(features_struct)
    current_features=features_struct{f_j};
    time_n=unique(current_features(:,end-1));
    s_current_features=[];
    for t_i=1:length(time_n)
        time_f_N=find(time_n(t_i)==current_features(:,end-1));
        if length(time_f_N) > t_f_size
            selected_f_N=time_f_N(randperm(length(time_f_N),t_f_size));
        else
            selected_f_N=time_f_N;
        end
        s_current_features=vertcat(s_current_features,current_features(selected_f_N,:));
        clear selected_f_N time_f_N
    end

    f24_tmp=s_current_features(s_current_features(:,end-1)<=24,:);
    f48_tmp=s_current_features(s_current_features(:,end-1)<=48 & s_current_features(:,end-1)>24,:);
    f72_tmp=s_current_features(s_current_features(:,end-1)<=72 & s_current_features(:,end-1)>48,:);

    features_all_24=[features_all_24;f24_tmp];
    features_all_48=[features_all_48;f48_tmp];
    features_all_72=[features_all_72;f72_tmp];


    f24_tmp_sub=[];
    f48_tmp_sub=[];
    f72_tmp_sub=[];

    for sub_method_i=1:length(median_bowl)-1

        if ~isempty(f24_tmp)
            f24_tmp_sub=[ f24_tmp_sub, sum(f24_tmp(:,sub_method_i) > median_bowl(sub_method_i) )/size(f24_tmp,1)];
        end

        if ~isempty(f48_tmp)
            f48_tmp_sub=[ f48_tmp_sub, sum(f48_tmp(:,sub_method_i) > median_bowl(sub_method_i) )/size(f48_tmp,1)];
        end

        if ~isempty(f72_tmp)
            f72_tmp_sub=[ f72_tmp_sub, sum(f72_tmp(:,sub_method_i) > median_bowl(sub_method_i) )/size(f72_tmp,1)];
        end

    end



    if ~isempty(f24_tmp)
        f24_tmp_sub=[ f24_tmp_sub,  f24_tmp(1,end)];
    end

    if ~isempty(f48_tmp)
        f48_tmp_sub=[ f48_tmp_sub,  f48_tmp(1,end)];
    end

    if ~isempty(f72_tmp)
        f72_tmp_sub=[ f72_tmp_sub,  f72_tmp(1,end)];
    end


    features_all_24_sub=[features_all_24_sub;f24_tmp_sub];
    features_all_48_sub=[features_all_48_sub;f48_tmp_sub];
    features_all_72_sub=[features_all_72_sub;f72_tmp_sub];



    clear current_features s_current_features
    clear f24_tmp f48_tmp f72_tmp full_tmp
end

features_all_24(:,end-1)=[];
features_all_48(:,end-1)=[];
features_all_72(:,end-1)=[];

fm_range_tmp{1}=1:T_D_L;
fm_range_tmp{2}=T_D_L+1:T_D_L+channel_L;
fm_range_tmp{3}=T_D_L+channel_L+1:T_D_L+channel_L*2;
fm_range_tmp{4}=T_D_L+channel_L*2+1:T_D_L+channel_L*3;
fm_range_tmp{5}=T_D_L+channel_L*3+1;
fm_range_tmp{6}=T_D_L+1+channel_L*3+1:T_D_L+1+channel_L*3+e_div_L*2;
fm_range_tmp{7}=T_D_L+1+channel_L*3+e_div_L*2+1:T_D_L+1+channel_L*3+e_div_L*4;
fm_range_tmp{8}=T_D_L+1+channel_L*3+e_div_L*4+1:T_D_L+1+channel_L*3+e_div_L*6;


fm_range{1}=[fm_range_tmp{1} fm_range_tmp{6}];
fm_range{2}=[fm_range_tmp{1} fm_range_tmp{6} fm_range_tmp{7}];
fm_range{3}=[fm_range_tmp{1} fm_range_tmp{7} fm_range_tmp{8}];
fm_range{4}=[fm_range_tmp{6} fm_range_tmp{7} fm_range_tmp{8}];
fm_range{5}=1:T_D_L+1+channel_L*3+e_div_L*6;

f_m_size=length(fm_range);
r_gp_size=5;
time_m_size=3;

model_outcome=[];
model_outcome_sub=[];

for m_i=1:time_m_size
    features_all=[];
    if m_i==1
        features_all=[features_all_24;features_all_48];
        features_all_sub=[features_all_24_sub;features_all_48_sub];
    elseif m_i==2
        features_all=[features_all_24;features_all_72];
        features_all_sub=[features_all_24_sub;features_all_72_sub];
    elseif m_i==3
        features_all=[features_all_24;features_all_48;features_all_72];
        features_all_sub=[features_all_24_sub;features_all_48_sub;features_all_72_sub];
    end

    if size(features_all,1) > 10
        good_Nums_all=find(features_all(:,end)==0);
        poor_Nums_all=find(features_all(:,end)==1);
        if length(good_Nums_all) > length(poor_Nums_all)
            feature_L_max=round(length(poor_Nums_all)*0.2);
        else
            feature_L_max=round(length(good_Nums_all)*0.2);
        end
        for r_gp_i=1:r_gp_size
            good_Nums_tmp(:,1)=unique(randperm(length(good_Nums_all),feature_L_max));
            poor_Nums_tmp(:,1)=unique(randperm(length(poor_Nums_all),feature_L_max));
            inp_f_Nums=[good_Nums_tmp; poor_Nums_tmp];
            for f_m_i=1:f_m_size
                model_outcome{m_i,f_m_i,r_gp_i} = fitcsvm(features_all(inp_f_Nums,fm_range{f_m_i}),features_all(inp_f_Nums,end),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
                model_outcome_sub{m_i,f_m_i,r_gp_i} = TreeBagger(10,features_all_sub(:,fm_range{f_m_i}),features_all_sub(:,end));
                clear f_m_i
            end
            clear inp_f_Nums good_Nums_tmp poor_Nums_tmp
        end
    end
    clear good_Nums_all poor_Nums_all
    clear features_all m_i
end

% model save
model_cpc=model_outcome;
% save_model(model_outcome,model_cpc,output_directory);
save_model(model_outcome,model_outcome_sub,model_cpc,output_directory);

end

%% functions
%---------------------------------------------
function save_model(model_outcome,model_outcome_sub,model_cpc,output_directory)
% Save results.
filename = fullfile(output_directory,'model.mat');
save(filename,'model_outcome','model_outcome_sub','model_cpc','-v7.3');
disp('Done.')
end

% function save_model(model_outcome,model_cpc,output_directory)
% % Save results.
% filename = fullfile(output_directory,'model.mat');
% save(filename,'model_outcome','model_cpc','-v7.3');
% disp('Done.')
% end


%---------------------------------------------
function outcome=get_outcome(patient_metadata)
patient_metadata=strsplit(patient_metadata,'\n');
outcome_tmp=patient_metadata(startsWith(patient_metadata,'Outcome:'));
outcome_tmp=strsplit(outcome_tmp{1},':');
if strncmp(strtrim(outcome_tmp{2}),'Good',4)
    outcome=0;
elseif strncmp(strtrim(outcome_tmp{2}),'Poor',4)
    outcome=1;
else
    keyboard
end
end

%---------------------------------------------
function [PartA, PartB]=z_bipartition_TE(channel_length)
G = 1:channel_length;
f = 1;
for N = 2:length(G)
    cases = nchoosek(G, N);
    maxM = N-1; %
    for idxD=1:size(cases,1)
        m = 0; %
        for idxC=1:maxM
            tmp = nchoosek(1:N, idxC);
            for idxS=1:size(tmp,1)

                m = m + 1;
                PartA{f,1}{idxD,m} = cases(idxD,tmp(idxS,:));
                PartB{f,1}{idxD,m} = setdiff(cases(idxD,:),PartA{f,1}{idxD,m});
            end
        end
    end
    f = f+1;
end
end


