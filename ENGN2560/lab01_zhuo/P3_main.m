%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab01: Camera Projection, Camera Calibration, and Image Features
%> Problem3: Image Features
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> ===========================================
%> TODO: Use VLFeat to detect SIFT features
%> ===========================================

image1 = imread('data\Problem3\a1.jpg');
image2 = imread('data\Problem3\a2.jpg');

image3 = imread('data\Problem3\b1.png');
image4 = imread('data\Problem3\b2.png');

if size(image1, 3) == 3
    image1 = rgb2gray(image1);
end

if size(image2, 3) == 3
    image2 = rgb2gray(image2);
end

if size(image3, 3) == 3
    image3 = im2gray(image3);
end

if size(image4, 3) == 3
    image4 = im2gray(image4);
end



PeakThresh = 1;

EdgeThresh = 10;

[f1, d1] = vl_sift(single(image1), 'PeakThresh', PeakThresh, 'EdgeThresh', EdgeThresh);
[f2, d2] = vl_sift(single(image2), 'PeakThresh', PeakThresh, 'EdgeThresh', EdgeThresh);

[f3, d3] = vl_sift(single(image3), 'PeakThresh', PeakThresh, 'EdgeThresh', EdgeThresh);
[f4, d4] = vl_sift(single(image4), 'PeakThresh', PeakThresh, 'EdgeThresh', EdgeThresh);

x1 = f1(1,:);
y1 = f1(2,:);
x2 = f2(1,:);
y2 = f2(2,:);
x3 = f3(1,:);
y3 = f3(2,:);
x4 = f4(1,:);
y4 = f4(2,:);

figure();
subplot(2, 2, 1);
imshow(image1);
title('Image 1');

subplot(2, 2, 2);
imshow(image2);
title('Image 2');

subplot(2, 2, 3);
imshow(image1);
hold on;
scatter(x1, y1, 3, 'og');

subplot(2, 2, 4);
imshow(image2);
hold on;
scatter(x2, y2, 3, 'sb');


figure();
subplot(2, 2, 1);
imshow(image3);
title('Image 3');

subplot(2, 2, 2);
imshow(image4);
title('Image 4');

subplot(2, 2, 3);
imshow(image3);
hold on;
scatter(x3, y3, 3, 'og');

subplot(2, 2, 4);
imshow(image4);
hold on;
scatter(x4, y4, 3, 'sb');

