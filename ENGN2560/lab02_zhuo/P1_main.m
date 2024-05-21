%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab02: Feature Correspondences, Camera Relative Pose, and RANSAC
%> Problem1: Camera Calibration from a Cube of Checkerboard
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Load 3D-2D correspondences 
%  (3D points are in the world coorindate; 2D points are in the image coordinate)
load("data/Problem1/cube_correspondences.mat");

%> ===================================================
%> TODO: Find camera intrinsic and extrinsic matrices
%> ===================================================

WP = points3D;
IP = points2D;
s = size(IP, 1);

a = [];
b = [];

for i=1:s
    var1 = WP(i,1);
    var2 = WP(i,2);
    var3 = WP(i,3);
    var4 = 1;
    var5 = 0;
    var6 = 0;
    var7 = 0;
    var8 = 0;
    var9 = -(WP(i,1))*IP(i,1);
    var10 = -(WP(i,2))*IP(i,1);
    var11 = -(WP(i,3))*IP(i,1);
    a_vec1 = [var1 var2 var3 var4 var5 var6 var7 var8 var9 var10 var11];
    a = cat(1,a,a_vec1);
    
    var1 = 0;
    var2 = 0;
    var3 = 0;
    var4 = 0;
    var5 = WP(i,1);
    var6 = WP(i,2);
    var7 = WP(i,3);
    var8 = 1;
    var9 = -(WP(i,1))*IP(i,2);
    var10 = -(WP(i,2))*IP(i,2);
    var11 = -(WP(i,3))*IP(i,2);
    a_vec2 = [var1 var2 var3 var4 var5 var6 var7 var8 var9 var10 var11];
    a = cat(1,a,a_vec2);

    b_vec1 = [IP(i,1)];
    b = cat(1,b,b_vec1);

    b_vec2 = [IP(i,2)];
    b = cat(1,b,b_vec2);

end

m = (a'*a)\(a'*b);

a1 = [m(1,1); m(2,1); m(3,1)];
a2 = [m(5,1); m(6,1); m(7,1)];
a3 = [m(9,1); m(10,1); m(11,1)];


lam = norm(a3);

xi = a1'*a3/norm(a3)^2;

eta = a2'*a3/norm(a3)^2;

beta = norm(cross(a2, a3))/norm(a3)^2;

sigma = (cross(a1, a3)' * cross(a2, a3))/(norm(cross(a2, a3))^2*norm(a3)^2);

alpha = sqrt((norm((cross(a1, a3)))^2/(norm(a3))^4) - sigma^2);

r1 = cross(a2, a3)/norm(cross(a2, a3));
r3 = a3/norm(a3);
r2 = cross(r3, r1);

R1 = [r1'; r2'; r3'];
K = [alpha, sigma, xi; 0, beta, eta; 0, 0, 1];
B = [m(4,1); m(8,1); 1];

T1 = (1/norm(a3))*K\B;

T2 = -T1;

R2 = [r1'; -r2'; -r3'];

o1 = 0;
o2 = 0;
o3 = 0;
o4 = 0;

for i = 1:s
    s1 = [R1, T1];
    s2 = [R1, T2];
    s3 = [R2, T1];
    s4 = [R2, T2];


    im1 = K*(R1*WP(i,:)' + T1);
    im1 = im1 ./ im1(end);
    
    d1 = sqrt((im1(1,1) - points2D(i,1))^2 + (im1(2,1) - points2D(i,2))^2);

    o1 = o1 + d1;

    im2 = K*(R1*WP(i,:)' + T2);
    im2 = im2 ./ im2(end);
    
    d2 = sqrt((im2(1,1) - points2D(i,1))^2 + (im2(2,1) - points2D(i,2))^2);

    o2 = o2 + d2;

    im3 = K*(R2*WP(i,:)' + T1);
    im3 = im3 ./ im3(end);
    
    d3 = sqrt((im3(1,1) - points2D(i,1))^2 + (im3(2,1) - points2D(i,2))^2);

    o3 = o3 + d3;

    im4 = K*(R2*WP(i,:)' + T2);
    im4 = im4 ./ im4(end);
    
    d4 = sqrt((im4(1,1) - points2D(i,1))^2 + (im4(2,1) - points2D(i,2))^2);

    o4 = o4 + d4;

end

disp('R1 with T1 reprojection error:');
disp(o1);
disp('R1 with T2 reprojection error:');
disp(o2);
disp('R2 with T1 reprojection error:');
disp(o3);
disp('R2 with T2 reprojection error:');
disp(o4);

o = [o1 o2 o3 o4];

[minValue, minIndex] = min(o);

if minIndex==1
    disp('R1 and T1 is a veridical camera extrinsic matrix');
elseif minIndex==2
    disp('R1 and T2 is a veridical camera extrinsic matrix');
elseif minIndex==3
    disp('R2 and T1 is a veridical camera extrinsic matrix');
else
    disp('R2 and T2 is a veridical camera extrinsic matrix');
end


