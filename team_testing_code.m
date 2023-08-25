%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Run trained classifier and obtain classifier outputs
% Inputs:
% 1. model
% 2. data directory
% 3. patient id
%
% Outputs:
% 1. outcome
% 2. outcome probability
% 3. CPC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outcome_binary, outcome_probability, cpc] = team_testing_code(model,input_directory,patient_id,verbose)

% transfer entropy - channel combination
channels = {'Fp1', 'Fp2', 'F3','F4'};
test_chs_L=3;
[PartA, PartB]=z_bipartition_TE(test_chs_L);

model_outcome=model.model_outcome;

e_div_L=4;
T_D_L=3;
channel_L=4;

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

time_m_size=3;

try    
    features=get_features(input_directory,patient_id,PartA,PartB,channels);
    fm_range{4}=1:size(features,2)-1;
    f_m_size=length(fm_range);

    if sum(sum(features))~=0
        if sum(sum(isnan(features')))~=0
            features(sum(isnan(features'))>0,:)=[];
        end
        if size(features,1)>0
            c_d_num=[];
            for c_i=1:size(features,2)-1
                d_tmp=find(abs(features(:,c_i)) > 999);
                c_d_num=vertcat(c_d_num,d_tmp);
                clear d_tmp
            end
            if sum(c_d_num)~=0
                features(c_d_num,:)=[];
            end
            if size(features,1)>0
                s_current_features=features;
                t24_tmp=s_current_features(s_current_features(:,end)<=24,1:end-1);
                t48_tmp=s_current_features(s_current_features(:,end)<=48 & s_current_features(:,end)>24,1:end-1);
                t72_tmp=s_current_features(s_current_features(:,end)<=72 & s_current_features(:,end)>48,1:end-1);
                clear s_current_features features

                outcome_p_all=ones(f_m_size,time_m_size)*-1;

                for m_i=1:time_m_size
                    if m_i==1
                        t_features=t24_tmp;
                    elseif m_i==2
                        t_features=[t24_tmp;t48_tmp];
                    elseif m_i==3
                        t_features=[t48_tmp;t72_tmp];
                    end

                    if sum(sum(t_features))~=0
                        if size(t_features,1)>0
                            decision_all=zeros(size(t_features,1),f_m_size);
                            for mf_i=1:f_m_size
                                model_solo=model_outcome{m_i,mf_i};
                                decision_all(:,mf_i)= predict(model_solo,t_features(:,fm_range{mf_i}));
                                clear model_solo
                            end
                            % decision_all
                            outcome_p_all(:,m_i)=mean(decision_all);
                        end
                        clear decision_all
                    end
                end


                % method별 weight
                outcome_w_all=outcome_p_all;
                outcome_w_all(1,:)=outcome_w_all(1,:)*0.24;
                outcome_w_all(2,:)=outcome_w_all(2,:)*0.25;
                outcome_w_all(3,:)=outcome_w_all(3,:)*0.25;
                outcome_w_all(4,:)=outcome_w_all(4,:)*0.26;
                outcome_all=sum(outcome_w_all);

                % 시간대별 weight
                avail_m=find(outcome_all(1,:) >= 0);
                if length(avail_m)==3
                    w_p=[0.3 0.4 0.3];
                elseif length(avail_m)==2
                    if length(intersect(avail_m,[1 2]))==2
                        w_p=[0.4 0.6 0];
                    elseif length(intersect(avail_m,[1 3]))==2
                        w_p=[0.5 0 0.5];
                    elseif length(intersect(avail_m,[2 3]))==2
                        w_p=[0 0.6 0.4];
                    end
                elseif length(avail_m)==1
                    if avail_m==1
                        w_p=[1 0 0];
                    elseif avail_m==2
                        w_p=[0 1 0];
                    elseif avail_m==3
                        w_p=[0 0 1];
                    end
                end
                outcome_probability=outcome_all*w_p';
            else
                outcome_probability=1; % poor
            end
        else
            outcome_probability=1; % poor
        end
    else
        outcome_probability=1; % poor
    end
    
    % outcome_binary
    if outcome_probability <= 0.5
        outcome_binary=0; % good
    else
        outcome_binary=1; % poor
    end
    % cpc
    if outcome_probability <= 0.5
        if outcome_probability <= 0.3
            cpc=1;
        else
            cpc=2;
        end
    else
        if outcome_probability <= 0.6
            cpc=3;
        elseif outcome_probability <= 0.7
            cpc=4;
        else
            cpc=5;
        end
    end
catch    
    outcome_probability=1; % poor
    outcome_binary=1; % poor
    cpc=5; % poor
end


% prob. modify
if outcome_probability==0
    outcome_probability=mod(randn(1,1),1)/100;
elseif outcome_probability==1
    outcome_probability=1-mod(randn(1,1),1)/100;
end


end


%---------------------------------------------
function [PartA, PartB]=z_bipartition_TE(channel_length)
G = 1:channel_length;
f = 1;
for N = 2:length(G)
    % N=Nneurons;
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