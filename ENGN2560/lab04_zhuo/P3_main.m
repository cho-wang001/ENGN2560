%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab04: Absolute Camera Pose and Visual Odometry
%> Problem3: Visual Odometry Part II
%> ------------------------------------------------------------------------
clc; clear all; close all;
%rng(0);

%> Load camera intrinsic matrix (borrowed from Problem2)
load('data/Problem2/IntrinsicMatrix.mat');

%> Load ground truth poses (borrowed from Problem2)
load('data/Problem2/GT_Poses.mat');	%> GT_Poses

%> Read all images in the sequence.
%> Use imread(Image_Sequence(i).name); to read image i
mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir([mfiledir, '/data/Problem2/fr2_desk/*.png']);

%> Parameters passed to RANSAC
PARAMS.INLIER_THRESH                          = 2;      %> 2 pixels
PARAMS.RANSAC_ITERATIONS                      = 2000;   %> Total number of RANSAC iterations
PARAMS.NUM_OF_FRAMES_FROM_LAST_KF             = 20;
PARAMS.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF = 0.5;

%> =================================================================
%> TODO: Implement a Visual Odometry Pipeline Reducing Motion Drift
%> =================================================================



f1 = Image_Sequence(1).folder;
n1 = Image_Sequence(1).name;

p1 = [f1 '\' n1];

Img1 = imread(p1);

f2 = Image_Sequence(2).folder;
n2 = Image_Sequence(2).name;

p2 = [f2 '\' n2];

Img2 = imread(p2);

if size(Img1, 3) == 3
    Img1 = rgb2gray(Img1);
end

if size(Img2, 3) == 3
    Img2 = rgb2gray(Img2);
end


[R12, T12, I12, cc1, cc2] = P1(PARAMS, Img1, Img2, K);

rep = [eye(3) [0 0 0 ]'];

rt = cat(2, R12, T12);

rep = cat(3, rep, rt);

s = size(GT_Poses, 3);

[pp1, d1, d2] = tran(PARAMS, rep(:, 1:3, 1), rep(:, 4, 1), R12, T12, cc1, cc2, K);

kf = 2;

kkk = [kf];

for i = 3:s

    f1 = Image_Sequence(kf).folder;
    n1 = Image_Sequence(kf).name;

    p1 = [f1 '\' n1];

    image1 = imread(p1);

    f2 = Image_Sequence(i).folder;
    n2 = Image_Sequence(i).name;

    p2 = [f2 '\' n2];

    image2 = imread(p2);

    
    %[~, ~, ~, cc1, cc2] = P1(PARAMS, image1, image2, K);
    if size(image1, 3) == 3
        image1 = rgb2gray(image1);
    end
    
    if size(image2, 3) == 3
        image2 = rgb2gray(image2);
    end
    
    [f1_, d1_] = vl_sift(single(image1));
    [f2_, d2_] = vl_sift(single(image2));
    
    [matches, scores] = vl_ubcmatch(d1_, d2_);
    %[~, sortedIndices] = sort(scores);
    %sortedMatches = matches(:, sortedIndices);
    
    cc1 = f1_(1:2, matches(1,:));
    cc2 = f2_(1:2, matches(2,:));
    cc1 = [cc1; ones(1,size(cc1,2))];
    cc2 = [cc2; ones(1,size(cc2,2))];
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    cc2s = size(cc2,2);

    [~, X, Y] = intersect(d2', cc1', "rows");
    cc1_no_3d_points = cc1';
    cc2_no_3d_points = cc2';

    pp1a = pp1(:,X);

    cc2 = cc2(:,Y);

    [AR, AT] = Ransac4AbsPose(PARAMS, pp1a, cc2, K);

    rt = cat(2, AR, AT);

    rep = cat(3, rep, rt);

    th = size(Y,1)/cc2s;

    if i == 22
        fprintf("here\n");
    end

    if th<=0.5 || (i-kf)>=20


        
        okf = kf;

        kf = i;

        kkk = [kkk, kf];
        cc1_no_3d_points(Y,:) = [];
        cc2_no_3d_points(Y,:) = [];
        cc1_no_3d_points = cc1_no_3d_points';
        cc2_no_3d_points = cc2_no_3d_points';

        [ppp1, dd1, dd2] = tran(PARAMS, rep(:,1:3,okf), rep(:,4,okf), rep(:,1:3,kf), rep(:,4,kf), cc1_no_3d_points, cc2_no_3d_points, K);

        pp1 = [pp1a, ppp1];

        d2 = [cc2, dd2];
        

    end

    i
end

R1_gt = GT_Poses(1:3,1:3,1);
T1_gt = GT_Poses(1:3,4,1);
C1_gt = -R1_gt * T1_gt;
R2_gt = GT_Poses(1:3,1:3,2);
T2_gt = GT_Poses(1:3,4,2);
C2_gt = -R2_gt * T2_gt;

scale = norm(C1_gt - C2_gt);

Estimated_Poses = rep;
for i = 1:size(Estimated_Poses,3)
    Estimated_Poses(:,4,i) = rep(:,4,i).*scale;
end

rep = Estimated_Poses;

RMSER = 0;

for i = 1:s
    r = rep(:,1:3,i);
    rg = GT_Poses(:, 1:3, i);
    re = (acos(0.5 * (trace(rg' * r) - 1)))^2;

    RMSER = RMSER + re;

end

RMSER = sqrt(RMSER/s);

RMSET = 0;

for i = 2:s
    t = rep(:,4,i);
    tg = GT_Poses(:, 4, i);
    tg = tg ./ norm(tg);
    t = t ./ norm(t);
    te = (abs(dot(tg, t) - 1))^2;

    RMSET = RMSET + te;
end

RMSET = sqrt(RMSET/s);


disp("RMSE for rotations");
disp(RMSER);

disp("RMSE for translations");
disp(RMSET);

Visualize_Trajectory(GT_Poses, Estimated_Poses, kkk);


function [P, c1, c2] = tran(PARAMS, R11, T11, R12, T12, cc1, cc2, K)
    
    R = R12*R11';
    T = T12 - R12*R11'*T11;

    E = skew(T) * R; 
    
    ff = inv(K)'*E*inv(K);

    c1 = [];
    c2 = [];

    for n = 1:size(cc1, 2)
                
        

        c13d = [cc1(1, n); cc1(2, n); 1];

        A = c13d(1, 1)*ff(1,1) + c13d(2, 1)*ff(1,2) + ff(1,3);
        B = c13d(1, 1)*ff(2,1) + c13d(2, 1)*ff(2,2) + ff(2,3);
        C = c13d(1, 1)*ff(3,1) + c13d(2, 1)*ff(3,2) + ff(3,3);



        distance = abs(A * cc2(1, n) + B * cc2(2, n) + C) / sqrt(A^2 + B^2);

        if distance<PARAMS.INLIER_THRESH
            c1 = [c1, cc1(:, n)];
            c2 = [c2, cc2(:, n)];
        end

    end
    
    ik = inv(K);
    s = size(c1, 2);
    LWP = [];
    
    for i=1:s
        r1 = ik*c1(:,i);
        r2 = ik*c2(:,i);
    
        A = [ skew(r1) * [eye(3) [0;0;0]]; skew(r2) * [R T]];
        [~, ~, V] = svd(A);
        lwp = V(:,end);
        lwp = lwp ./ lwp(end);

        lwp = lwp(1:3, 1);

        %lwp = R11*lwp + T11;
        lwp = R11'*(lwp - T11);
    
        LWP = [LWP, lwp(1:3,1)];
    end
    P = LWP;
end


function [AbsR, AbsT] = Ransac4AbsPose(PARAMS, Points3D, Points2D,K)
    I = PARAMS.RANSAC_ITERATIONS;
    T = PARAMS.INLIER_THRESH;
    nm = size(Points3D, 2);
    max_inlier_count = 0;

    Points2D = Points2D(1:2,:);

    for i = 1:I
        
        sc = randperm(nm, 3);

        rn2 = inv(K)*[Points2D(:,sc); 1, 1, 1];

        rn3 = Points3D(:,sc);
        
        [rs, ts] = P3P_LambdaTwist(rn2, rn3);
        
        sss = size(rs, 3);
        %inlier_count = 0;
        
        for j = 1:sss
            rt = rs(:,:,j);
            tt = ts(:,j);
            inlier_count = 0;

            ee = [];
            for f = 1:size(Points3D,2)
                rp2d = rp(rt, tt, K, Points3D(:,f));

                e = sqrt((rp2d(1) - Points2D(1,f))^2 + (rp2d(2) - Points2D(2,f))^2);
                if e < PARAMS.INLIER_THRESH
                    inlier_count = inlier_count + 1;
                    ee = [ee, f];
                end
            end

            if inlier_count > max_inlier_count
                max_inlier_count = inlier_count;
                ir = rt;
                it = tt;
            end

        end
        
    end

    AbsR = ir;
    AbsT = it;
end



function S = skew(v)

    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end

function r = rp(R, T, K, g)
    ga = R*g +T;
    c = K*ga;
    c = c./c(end);

    r = c;
end
