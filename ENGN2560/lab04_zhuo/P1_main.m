%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab04: Absolute Camera Pose and Visual Odometry
%> Problem1: Absolute Pose Estimation
%> ------------------------------------------------------------------------
clc; clear all; close all;
% rng(0);

%> Read triplet image pair
Img1 = imread('data/Problem1/Img1.png');
Img2 = imread('data/Problem1/Img2.png');
Img3 = imread('data/Problem1/Img3.png');

%> Load camera intrinsic matrix
load('data/Problem1/IntrinsicMatrix.mat');

%> Load ground truth pose for the third image
load('data/Problem1/GT_R.mat');		%> R_gt
load('data/Problem1/GT_T.mat');		%> T_gt

%> Parameters passed to RANSAC
PARAMS.INLIER_THRESH                 = 2;      %> 2 pixels
PARAMS.RANSAC_ITERATIONS             = 2000;   %> Total number of RANSAC iterations

%> =========================================================
%> TODO: Estimate an Absolute Pose under a RANSAC scheme
%> =========================================================

[R12, T12, I12, cc1, cc2] = P1(PARAMS, Img1, Img2, K);
 
ik = inv(K);
s = size(cc1, 2);
LWP = [];

for i=1:s
    r1 = ik*cc1(:,i);
    r2 = ik*cc2(:,i);

    A = [ skew(r1) * [eye(3) [0 0 0]']; skew(r2) * [R12 T12]];
    [~, ~, V] = svd(A);
    lwp = V(:,end);
    lwp = lwp ./ lwp(end);

    LWP = [LWP, lwp(1:3,1)];
end

if size(Img1, 3) == 3
    Img1 = rgb2gray(Img1);
end

if size(Img2, 3) == 3
    Img2 = rgb2gray(Img2);
end

if size(Img3, 3) == 3
    Img3 = rgb2gray(Img3);
end

[f2, d2] = vl_sift(single(Img2));
[f3, d3] = vl_sift(single(Img3));

[matches, scores] = vl_ubcmatch(d2, d3);
[~, sortedIndices] = sort(scores);
sortedMatches = matches(:, sortedIndices);
mf2 = f2(1:2, sortedMatches(1,:));
mf3 = f3(1:2, sortedMatches(2,:));



[~, X, Y] = intersect(cc2(1:2,:)', mf2', 'rows');

Point3D = LWP(:,X);
Point2D = mf3(:,Y);


[abr, abt] = Ransac4AbsPose(PARAMS, Point3D, Point2D, K);

rotation_error = acos(0.5 * (trace(R_gt' * abr) - 1));
T_gt = T_gt ./ norm(T_gt);
abt = abt ./ norm(abt);
translation_error = abs(dot(T_gt, abt) - 1);
disp("rotation_error:");
disp(rotation_error);
disp("translation_error:");
disp(translation_error);

[R23, T23, ~, ~, ~] = P1(PARAMS, Img2, Img3, K);

R13 = R23 * R12;
T13 = R23 * T12 + T23;

rotation_errorl = acos(0.5 * (trace(R_gt' * R13) - 1));
T_gt = T_gt ./ norm(T_gt);
T13 = T13 ./ norm(T13);
translation_errorl = abs(dot(T_gt, T13) - 1);
disp("rotation_error (propogate):");
disp(rotation_errorl);
disp("translation_error (propogate):");
disp(translation_errorl);

function [AbsR, AbsT] = Ransac4AbsPose(PARAMS, Points3D, Points2D,K)
    I = PARAMS.RANSAC_ITERATIONS;
    T = PARAMS.INLIER_THRESH;
    nm = size(Points3D, 2);
    max_inlier_count = 0;

    for i = 1:I
        
        sc = randperm(nm, 3);

        rn2 = inv(K)*[Points2D(:,sc); 1, 1, 1];

        rn3 = Points3D(:,sc);
        
        [rs, ts] = P3P_LambdaTwist(rn2, rn3);
        
        sss = size(rs, 3);
        inlier_count = 0;
        
        for j = 1:sss
            rt = rs(:,:,j);
            tt = ts(:,j);

            for f = 1:size(Points3D,2)
                rp2d = rp(rt, tt, K, Points3D(:,f));

                e = sqrt((rp2d(1) - Points2D(1,f))^2 + (rp2d(2) - Points2D(2,f))^2);
                if e < PARAMS.INLIER_THRESH
                    inlier_count = inlier_count + 1;
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