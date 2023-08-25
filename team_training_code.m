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
    patient_id=patient_ids{j}
    
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

%% train model
e_div_L=4;
T_D_L=3;
channel_L=4;

model_outcome=[];
features_all_72=[];
features_all_48=[];
features_all_24=[];

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

    clear current_features s_current_features
    clear f24_tmp f48_tmp f72_tmp full_tmp
end

features_all_24(:,end-1)=[];
features_all_48(:,end-1)=[];
features_all_72(:,end-1)=[];

% TE
fm_range_tmp{1}=1:T_D_L;
% power
fm_range_tmp{2}=T_D_L+1:T_D_L+channel_L;
% SE_div
fm_range_tmp{3}=T_D_L+channel_L+1:T_D_L+channel_L+e_div_L*3;
% SE_origin
fm_range_tmp{4}=T_D_L+channel_L+e_div_L*3+1:T_D_L+channel_L+e_div_L*6;
% SE power
fm_range_tmp{5}=T_D_L+channel_L+e_div_L*6+1:T_D_L+channel_L+e_div_L*9;

fm_range{1}=[fm_range_tmp{1} fm_range_tmp{3}];
fm_range{2}=[fm_range_tmp{1} fm_range_tmp{3} fm_range_tmp{4}];
fm_range{3}=[fm_range_tmp{1} fm_range_tmp{4} fm_range_tmp{5}];
fm_range{4}=1:size(features_all_72,2)-1;

f_m_size=length(fm_range);
time_m_size=3;

for m_i=1:time_m_size    
    if m_i==1
        features_all=features_all_24;
    elseif m_i==2
        features_all=[features_all_24;features_all_48];
    elseif m_i==3
        features_all=[features_all_48;features_all_72];
    end

    if size(features_all,1) > 10
        for f_m_i=1:f_m_size
            model_outcome{m_i,f_m_i} = fitcsvm(features_all(:,fm_range{f_m_i}),features_all(:,end),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        end
    end

    clear features_all
end

% model save
model_cpc=model_outcome;
save_model(model_outcome,model_cpc,output_directory);

end

%% functions
%---------------------------------------------
function save_model(model_outcome,model_cpc,output_directory) 
% Save results.
filename = fullfile(output_directory,'model.mat');
save(filename,'model_outcome','model_cpc','-v7.3');
disp('Done.')
end

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


