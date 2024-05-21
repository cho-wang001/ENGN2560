%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab02: Feature Correspondences, Camera Relative Pose, and RANSAC
%> Problem3: Measure Outlier Ratio from Known Relative Camera Pose
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Load camera intrinsic matrix
load('data/Problem3/IntrinsicMatrix.mat');

%> Load relative rotation and translation 
load('data/Problem3/RelativeRotation.mat');
load('data/Problem3/RelativeTranslation.mat');

%> Load two images
Img1 = imread('data/Problem2/1.png');
Img2 = imread('data/Problem2/2.png');

%> =====================================================
%> TODO: Calculate outlier ratio and draw epipolar lines
%> =====================================================

t_cross = [0 -T(3) T(2); T(3) 0 -T(1); -T(2) T(1) 0];

E = t_cross*R;

invk = inv(K);

F = invk'*E*invk;

if size(Img1, 3) == 3
    Img1 = rgb2gray(Img1);
end

if size(Img2, 3) == 3
    Img2 = rgb2gray(Img2);
end

[f1, d1] = vl_sift(single(Img1));
[f2, d2] = vl_sift(single(Img2));

x1 = f1(1,:);
y1 = f1(2,:);
x2 = f2(1,:);
y2 = f2(2,:);

mf1 = [x1; y1];
mf2 = [x2; y2];

matches = vl_ubcmatch(d1, d2);

distances = sqrt(sum((d1(:, matches(1, :)) - d2(:, matches(2, :))).^2));
[~, sortedIndices] = sort(distances);
sortedMatches = matches(:, sortedIndices);

mf1 = mf1(:, sortedMatches(1,:));
mf2 = mf2(:, sortedMatches(2,:));

s = size(mf1, 2);

L = [];

for i = 1:s
    l = F*[mf1(1,i),mf1(2,i),1]';
    L = cat(2,L,l);
end

ol = 0;

for i = 1:s
    a = L(1, i);
    b = L(2, i);
    c = L(3, i);
    
    d = [];

    for j = 1:s
        distance = abs(a * mf2(1, j) + b * mf2(2, j) + c) / sqrt(a^2 + b^2);
        d = cat(2, d , distance);
    end

    [value, ~] = min(d);

    if value>=2
        ol = ol+1;
    end

end

or = ol/s;

disp('Outlier Ratio:');
disp(or);

figure();
I12 = [Img1, Img2];
imshow(I12);
hold on;

ss = size(I12, 2)./2;

for i = 1:20:60
    a = L(1, i);
    b = L(2, i);
    c = L(3, i);

    x = 1:ss;
    y = (-a * x - c) / b;
    plot((x + ss), y, 'g', 'LineWidth', 1);

    scatter(mf1(1,i),mf1(2,i), 'b', 'LineWidth', 1);

    pd = [];

    for j = 1:s
        distance = abs(a * mf2(1, j) + b * mf2(2, j) + c) / sqrt(a^2 + b^2);
        pd = cat(2, pd , distance);
    end

    [~, index] = min(pd);
    
    scatter((mf2(1,index)+ss),mf2(2,index), 'b', 'LineWidth', 1);

end

