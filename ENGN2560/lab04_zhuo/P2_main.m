%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab04: Absolute Camera Pose and Visual Odometry
%> Problem2: Visual Odometry Part I
%> ------------------------------------------------------------------------
clc; clear all; close all;
% rng(0);

%> Load camera intrinsic matrix
load('data/Problem2/IntrinsicMatrix.mat');

%> Load ground truth poses
load('data/Problem2/GT_Poses.mat');	%> GT_Poses

%> Read all images in the sequence.
%> Use imread(Image_Sequence(i).name); to read image i
mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir([mfiledir, '/data/Problem2/fr2_desk/*.png']);

%> Parameters passed to RANSAC
PARAMS.INLIER_THRESH                 = 2;      %> 2 pixels
PARAMS.RANSAC_ITERATIONS             = 1200;    %> Total number of RANSAC iterations

%> =========================================================
%> TODO: Implement a Naive Visual Odometry 
%> =========================================================

s = size(GT_Poses, 3);

rep = [eye(3) [0 0 0 ]'];


for i = 2:s
    GR = GT_Poses(:, 1:3, (i-1));
    GT = GT_Poses(:, 4, (i-1));

    GRi = GT_Poses(:, 1:3, i);
    GTi = GT_Poses(:, 4, i);

    ss = norm(-GRi'*GTi - (-GR'*GT));
    
    f1 = Image_Sequence(i - 1).folder;
    n1 = Image_Sequence(i - 1).name;

    p1 = [f1 '\' n1];

    image1 = imread(p1);

    f2 = Image_Sequence(i).folder;
    n2 = Image_Sequence(i).name;

    p2 = [f2 '\' n2];

    image2 = imread(p2);

    [RR, RT] = P2(PARAMS, image1, image2, K);

    RT = RT.*ss;

    RT = (RR*rep(:,4,i-1)) + RT;

    RR = RR*rep(:,1:3,i-1);

    rt = cat(2, RR, RT);

    rep = cat(3, rep, rt);

    disp(i-1);
end


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



Visualize_Trajectory(GT_Poses, rep, 0:30:180);



