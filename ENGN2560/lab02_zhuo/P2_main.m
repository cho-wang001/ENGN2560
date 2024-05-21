%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab02: Feature Correspondences, Camera Relative Pose, and RANSAC
%> Problem2: Feature Correspondences and Rank-Ordered List
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Read an image pair
Img1 = imread('data/Problem2/1.png');
Img2 = imread('data/Problem2/2.png');

%> =====================================================
%> TODO: Use VLFeat to find feature correspondences and 
%        construct a rank-ordered list
%> =====================================================

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

draw_Feature_Matches(Img1, Img2, mf1, mf2, 30, 1);