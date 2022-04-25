%-------------------------------------------------------------------------------
%   Adaptive Invariant EKF for map-aided localization using 3D point cloud
%   Author  : Zhongxing Tao
%   Date    : 2020-06-30 @ XJTU
%-------------------------------------------------------------------------------
clc; close all; clear all;
addpath('InvariantKF');
params = getParamFun();
run('ReadKitti00Map.m');
% File path for Kitti data
VelRoot = 'D:\KittiData\data_odometry_velodyne\sequences';
%
HDLFolder = sprintf('%s/%02d/velodyne/', VelRoot, params.SeqIdx);
SaveFolder = sprintf('ExpInEKFKitSeq%02d', params.SeqIdx);
%%
ssDate = '2023'; 
params.isAdp = 1;  
params.pointNum = 2000;
params.isGridDown = 0;
for Sel = 1 : 1 : 1
    vMethods = {'p2plICP', 'NDT', 'p2pICP', 'HMRF', 'MiNoM',  'SpsICP', 'IRLS'};
    C = 2000;  % [1000 - 20000]
    for nc = 1 : 1 : length(C)
        params.modelC = C(nc);
        params.RegM = vMethods{Sel};
        tTime = [];
        tRmse = [];
        tRatio= [];
        tNum  = [];
        TTf = params.GpsTf; % if isMatMap == 1
        TTf = [eul2rotm(rotm2eul(TTf(1:3,1:3))),TTf(1:3,end);0 0 0 1];
        % EKF
        Xk_1 = TTf(:,:,end);    % InitPose   [R,t]       % 2019-09-07
        Ck_1 = diag([0.001 0.001 0.001 0.001 0.001 0.001] .^ 2); % state = [ax ay az x y z]
        vCov = Ck_1;
        vehvX = CTF2Pose(Xk_1)';
        %
        params.isShow = 0;
        if params.isShow == 1
            HF = figure;hold on;grid on;
            set(gcf,'Position',[0 0 800 400], 'color','k');
            set(gca,'Position',[0.01 0.01 0.99,0.99], 'color','k');
        end
        params.isFirst = 1;
        params.errorS = eye(4); % 2021-03-25
        for id = params.StartID : params.dFrm : length( params.SelRange ) %  only for kitti 0426
            nFrm = params.SelRange(id) - 1;
            [ptHdl,ptOrg,~] = KitHdlReadRaw(nFrm, HDLFolder, params.pointNum, params.RadiusRange, ~params.isGridDown); % 0218 random sample
            tic;
            [XbkTF, Gx, Gu] = InEKF_Propagation( id, params.OdometryInfo, Xk_1, params );
            Cbk = Gx * Ck_1 * Gx' + Gu * params.CovM * Gu';
            %%
            pos = XbkTF(1:3,end)';
            vIdx = rangesearch(kdMap, pos, params.MaxR+10);
            subMap = pointCloud(ptMap.Location(vIdx{:},:));
            subMap.Normal = ptMap.Normal(vIdx{:},:);
            subkdMap = createns(subMap.Location);
            params.CorresTF = CalCorrespondTF(subkdMap, subMap, ptOrg, XbkTF, params);
            %%
            [Score, vNewR, vCstCov, vNewVQ1, vNewVQ2, vNewVQ3, vNewVQ4] = GeneTrainDataUsingP2PL(subkdMap, subMap, ptOrg, params);
            if params.isAdp == 1
                GpCov = TestGP4Q(params.K, vNewR, vNewVQ1, vNewVQ2, vNewVQ3, vNewVQ4, params.vRads); % new version dR = 2;2020-10-20 about 1 ms
                params.Dynamic = GpCov; % vRad + 4 quat
                lamda = 1;
                vTmpCov1 = lamda .* [params.Dynamic(:,2),params.Dynamic(:,2),params.Dynamic(:,2)] +  params.MeaNoise(:,2:4);
                vTmpCov2 = lamda .* [params.Dynamic(:,3),params.Dynamic(:,3),params.Dynamic(:,3)] +  params.MeaNoise(:,2:4);
                vTmpCov3 = lamda .* [params.Dynamic(:,4),params.Dynamic(:,4),params.Dynamic(:,4)] +  params.MeaNoise(:,2:4);
                vTmpCov4 = lamda .* [params.Dynamic(:,5),params.Dynamic(:,5),params.Dynamic(:,5)] +  params.MeaNoise(:,2:4);
                params.QuatCov = params.modelC .* [vTmpCov1, vTmpCov2, vTmpCov3, vTmpCov4];
                params.Cov = [GpCov(:,1), params.MeaNoise(:,2:4)];
            else % standard method
                params.Cov = [0, params.StdNoise];  % 2020-09-28
                params.Cov(:,2:4) = params.modelC .* params.Cov(:,2:4); % only for standard EKF
            end
            %
            [Xk, Ck] = InEKF_Update4Q(subkdMap, subMap, ptHdl, Cbk, XbkTF, params);
            %%
            Ck_1 = Ck;
            Xk_1 = Xk;
            vCov = [vCov;Ck_1];  %
            VecTF = Xk * params.InvCalTF;
            vehvX = [vehvX; CTF2Pose(VecTF)'];
            %%
            GpsT = params.vGrdTF(1:3,end,id);
            EstT = VecTF(1:3,end);
            if Score(1) < 0.0001 || norm(GpsT-EstT) > 10
                btest = 1;
                break;
            end
            
            tRatio(end+1,:) = Score(1);
            TTf(:, :, end+1) = Xk;
            tTime(end+1) = toc;
            tNum = [tNum; ptHdl.Count];
            %% --------------------------------------------------------------
            str0 = sprintf('%s, C = %d: Frame = %04d/%04d, PtsNum = %06d/%06d, Ratio = [%.4f,%.4f], Time = %04dms',params.RegM, params.modelC ,id, length(params.SelRange),...
                ptHdl.Count, ptMap.Count, Score(1), Score(2), ceil(1000.0*tTime(end)));
            disp(str0);
            params.isFirst = 2;
            
            if params.isShow == 1 
                run('ShowFun2.m');
                pause(1);
            end
        end
        vFrmInfor.Comparemethod = params.RegM;
        vFrmInfor.HdlDownRatio = params.pointNum;
        vFrmInfor.DataRoot = HDLFolder;
        vFrmInfor.tTime = tTime;
        vFrmInfor.tRatio = tRatio;
        vFrmInfor.vX = vehvX;       % very important!!!!!!!!!
        vFrmInfor.vCov = vCov;
        vFrmInfor.tNum = tNum;
        DataSave = fullfile(pwd, SaveFolder);
        if params.isGridDown == 1
            FileName = sprintf('%s/%sUInEKF%sC%dN%d.mat',DataSave, params.RegM, ssDate,C(nc),params.pointNum);
        else
            FileName = sprintf('%s/%sRInEKF%sC%dN%d.mat',DataSave, params.RegM, ssDate,C(nc),params.pointNum);
        end
        save(FileName,'vFrmInfor');
    end
end
