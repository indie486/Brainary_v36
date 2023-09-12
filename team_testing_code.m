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
model_outcome_sub=model.model_outcome_sub;

e_div_L=4;
T_D_L=3;
channel_L=4;

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

    
median_bowl=[0.0135834774755276	0.0534925919728622	0.118902167100248	36.9485539831741	39.8491382433925	20.4846472399850	23.6807960176608	8.11328125000000	8.25781250000000	8.91796875000000	9.10644531250000	9.70261436570684	9.70324585022036	9.29379984973723	9.32579317624609	0.937136158710965	2.18978102189781	2.70270270270270	3.50877192982456	0	1.26115180025903	1.32563204137101	0.980392156862745	0	1.61380235172861	1.65266917916043	1.65178901640987	1.54433412633408	1.44306693780431	1.47260015294299	1.45285761779823	1.29919958337640	0.0898696159686560	0.0908590021983754	0.0858913655268553	0.0744926384807600	0.0543367764720562	0.0545033744349376	0.0524064794530345	0.0459832914457559];


try    
    features=get_features(input_directory,patient_id,PartA,PartB,channels);    
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
                        t_features=[t24_tmp;t48_tmp];
                    elseif m_i==2
                        t_features=[t24_tmp;t72_tmp];
                    elseif m_i==3
                        t_features=[t24_tmp;t48_tmp;t72_tmp];
                    end


                    features_sub=[];

                    for sub_method_i=1:length(median_bowl)
                        features_sub=[ features_sub, sum(t_features(:,sub_method_i) > median_bowl(sub_method_i) )/size(t_features,1)];
                    end

                    break_on=0;
                    outcome_p_all_tmp=[];
                    if sum(sum(t_features))~=0
                        if size(t_features,1)>0
                            for r_gp_i=1:r_gp_size

                                decision_all=zeros(size(t_features,1),f_m_size);
                                decision_all_sub=zeros(size(t_features,1),f_m_size);

                                for mf_i=1:f_m_size
                                    model_solo=model_outcome{m_i,mf_i,r_gp_i};
                                    if ~isempty(model_solo)
                                        decision_all(:,mf_i)= predict(model_solo,t_features(:,fm_range{mf_i}));
                                    else
                                        break_on=1;
                                        break
                                    end

                                    model_solo_sub=model_outcome_sub{m_i,mf_i,r_gp_i};
                                    if ~isempty(model_solo_sub)
                                        [bi,prob_sub]=model_solo_sub.predict(features_sub(:,fm_range{mf_i}));
                                        decision_all_sub(:,mf_i)=prob_sub(:,2);
                                    end
                                    clear model_solo model_solo_sub
                                end
                                
                                if break_on==0
                                    outcome_p_all_tmp(:,r_gp_i)=mean(decision_all);
                                else
                                    break
                                end
                            end
                        end
                        clear decision_all
                        outcome_p_all(:,m_i)=mean(outcome_p_all_tmp');

                        outcome_p_all_sub(:,m_i)=mean(decision_all_sub);
                    end
                    clear t_features outcome_p_all_tmp
                end


                % method별 weight
                outcome_w_all=outcome_p_all;
                outcome_w_all(1,:)=outcome_w_all(1,:)*0.2;
                outcome_w_all(2,:)=outcome_w_all(2,:)*0.15;
                outcome_w_all(3,:)=outcome_w_all(3,:)*0.15;
                outcome_w_all(4,:)=outcome_w_all(4,:)*0.15;
                outcome_w_all(5,:)=outcome_w_all(5,:)*0.35;
                outcome_all=sum(outcome_w_all);


                outcome_w_all_sub=outcome_p_all_sub;
                outcome_w_all_sub(1,:)=outcome_w_all_sub(1,:)*0.2;
                outcome_w_all_sub(2,:)=outcome_w_all_sub(2,:)*0.15;
                outcome_w_all_sub(3,:)=outcome_w_all_sub(3,:)*0.15;
                outcome_w_all_sub(4,:)=outcome_w_all_sub(4,:)*0.15;
                outcome_w_all_sub(5,:)=outcome_w_all_sub(5,:)*0.35;
                outcome_all_sub=sum(outcome_w_all_sub);

                % 시간대별 weight
                avail_m=find(outcome_all(1,:) >= 0);
                if length(avail_m)==3
                    w_p=[0.3 0.2 0.5];
                elseif length(avail_m)==2
                    if length(intersect(avail_m,[1 2]))==2
                        w_p=[0.6 0.4 0];
                    elseif length(intersect(avail_m,[1 3]))==2
                        w_p=[0.3 0 0.7];
                    elseif length(intersect(avail_m,[2 3]))==2
                        w_p=[0 0.2 0.8];
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
                out_probability_sub=outcome_all_sub*w_p';
                out_probability_final=outcome_probability;
                out_probability_final(outcome_probability < 0.5 & out_probability_sub >0.8)=out_probability_sub;

                outcome_probability=out_probability_final;


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
