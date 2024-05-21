%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab03: 3D Reconstruction and Bundle Adjustment
%> Problem1: Reconstruction by Triangulation
%> ------------------------------------------------------------------------
clc; clear all; close all;
% rng(0);

%> Read an image pair
Img1 = imread('data/Problem1/Img1.png');
Img2 = imread('data/Problem1/Img2.png');

%> Additional image
Img3 = imread('data/Problem1/Img3.png');

%> Load camera intrinsic matrix
load('data/Problem1/IntrinsicMatrix.mat');

%> Parameters passed to RANSAC
PARAMS.INLIER_THRESH                 = 2;       %> 2 pixels
PARAMS.RANSAC_ITERATIONS             = 2500;    %> Total number of RANSAC iterations
PARAMS.SCALE                         = 0.29;    %> Scale of the scene for 2-view triangulation
PARAMS.GT_PLANE_NORMAL_VECTOR        = [-0.0003, 0.5792, 0.8152];
PARAMS.GT_PLANE_DISPLACEMENT         = -1.0545;

%> =========================================================
%> TODO: Implement and Evaluate Three Triangulation Methods
%> =========================================================

[R12, T12, I12, cc1, cc2] = P1(PARAMS, Img1, Img2, K);

ik = inv(K);

s = size(cc1, 2);

SWP = [];
LWP = [];

c1 = -eye(3)'*[0 0 0]';
c2 = -R12'*T12;

for i=1:s
    r1 = ik*cc1(:,i);
    r2 = ik*cc2(:,i);

    A = (eye(3) - (eye(3)'*r1*r1'*eye(3))./(norm(r1)^2)) + (eye(3) - (R12'*r2*r2'*R12)./(norm(r2)^2));
    b = (eye(3) - (eye(3)'*r1*r1'*eye(3))./(norm(r1)^2))*c1 + (eye(3) - (R12'*r2*r2'*R12)./(norm(r2)^2))*c2;
    swp = A \ b;

    SWP = [SWP, swp];
end

for i=1:s
    r1 = ik*cc1(:,i);
    r2 = ik*cc2(:,i);
    
    A = [ skew(r1) * [eye(3) [0 0 0]']; skew(r2) * [R12 T12]];
    
    [~, ~, V] = svd(A);

    lwp = V(:,end);
    lwp = lwp ./ lwp(end);

    LWP = [LWP, lwp(1:3,1)];
end

OC1 = [];
OC2 = [];

skew_T = [0, -T12(3), T12(2); T12(3), 0, -T12(1); -T12(2), T12(1), 0];

E = skew_T * R12;

for i = 1:s
    trd = LWP(:,i);

    r1 = rp(eye(3), [0 0 0]', K, trd);
    r2 = rp(R12, T12, K, trd);

    X = [r1(1,1), r1(2,1), r2(1,1), r2(2,1), 1];

    Energy = @(x) (evaluateEnergyFunction(x, cc1(1:2,i)', cc2(1:2,i)', E));
    options = optimoptions('lsqnonlin', 'Display', 'iter');
    options.Algorithm = 'levenberg-marquardt';

    optimal_points = lsqnonlin(Energy, X, [], [], options);


    oc1 = optimal_points(1:2);
    oc2 = optimal_points(3:4);

    OC1 = cat(2, OC1, [oc1 1]');
    OC2 = cat(2, OC2, [oc2 1]');
end


KLWP = [];

for i=1:s
    r1 = ik*OC1(:,i);
    r2 = ik*OC2(:,i);

    A = [ skew(r1) * [eye(3) [0 0 0]']; skew(r2) * [R12 T12]];

    [~, ~, V] = svd(A);

    klwp = V(:,end);
    klwp = klwp ./ klwp(end);

    KLWP = [KLWP, klwp(1:3,1)];
end

SSWP = SWP.*PARAMS.SCALE;
SLWP = LWP.*PARAMS.SCALE;
SKLWP = KLWP.*PARAMS.SCALE;

DSWP = [];

for i=1:size(SSWP,2)
    dswp = abs(SSWP(:,i)' * PARAMS.GT_PLANE_NORMAL_VECTOR' + PARAMS.GT_PLANE_DISPLACEMENT) / norm(PARAMS.GT_PLANE_NORMAL_VECTOR);
    DSWP = [DSWP, dswp];
end

figure();
h1 = histogram(DSWP);
h1.BinWidth = 0.005;
title('Distance Distribution to Ground Truth Plane (Symmedian)');
xlabel('Distance to Plane');
ylabel('Frequency');

DSLWP = [];

for i=1:size(SLWP,2)
    lswp = abs(SLWP(:,i)' * PARAMS.GT_PLANE_NORMAL_VECTOR' + PARAMS.GT_PLANE_DISPLACEMENT) / norm(PARAMS.GT_PLANE_NORMAL_VECTOR);
    DSLWP = [DSLWP, lswp];
end

figure();
h2 = histogram(DSLWP);
h2.BinWidth = 0.005;
title('Distance Distribution to Ground Truth Plane (Linear)');
xlabel('Distance to Plane');
ylabel('Frequency');

DSKLWP = [];

for i=1:size(SKLWP,2)
    kswp = abs(SKLWP(:,i)' * PARAMS.GT_PLANE_NORMAL_VECTOR' + PARAMS.GT_PLANE_DISPLACEMENT) / norm(PARAMS.GT_PLANE_NORMAL_VECTOR);
    DSKLWP = [DSKLWP, kswp];
end

figure();
h2 = histogram(DSKLWP);
h2.BinWidth = 0.005;
title('Distance Distribution to Ground Truth Plane (Kanatani)');
xlabel('Distance to Plane');
ylabel('Frequency');



%2 to 3 views
[R13, T13, I13, vv1, vv2] = P1(PARAMS, Img1, Img3, K);

[~ , f2, f3] = intersect(cc1', vv1', "rows");

for i = f2
    c231 = cc1(:,i);
end

for i = f2
    c232 = cc2(:,i);
end

for i = f3
    c233 = vv2(:,i);
end


c3 = -R13'*T13;

s3 = size(c231,2);

SWP3 = [];

for i=1:s3
    r1 = ik*c231(:,i);
    r2 = ik*c232(:,i);
    r3 = ik*c233(:,i);

    A = (eye(3) - (eye(3)'*r1*r1'*eye(3))./(norm(r1)^2)) + (eye(3) - (R12'*r2*r2'*R12)./(norm(r2)^2)) + (eye(3) - (R13'*r3*r3'*R13)./(norm(r3)^2));
    b = (eye(3) - (eye(3)'*r1*r1'*eye(3))./(norm(r1)^2))*c1 + (eye(3) - (R12'*r2*r2'*R12)./(norm(r2)^2))*c2 + (eye(3) - (R13'*r3*r3'*R13)./(norm(r3)^2))*c3;
    
    swp = A \ b;

    SWP3 = [SWP3, swp];
end

SSWP3 = SWP3.* 0.625;

DSWP3 = [];

for i=1:size(SSWP3,2)
    dswp = abs(SSWP3(:,i)' * PARAMS.GT_PLANE_NORMAL_VECTOR' + PARAMS.GT_PLANE_DISPLACEMENT) / norm(PARAMS.GT_PLANE_NORMAL_VECTOR);
    DSWP3 = [DSWP3, dswp];
end

figure();
h1 = histogram(DSWP3);
h1.BinWidth = 0.005;
title('Distance Distribution to Ground Truth Plane (Symmedian 3 Views)');
xlabel('Distance to Plane');
ylabel('Frequency');


LWP3 = [];

for i=1:s3

    r1 = ik*c231(:,i);
    r2 = ik*c232(:,i);
    r3 = ik*c233(:,i);
    
    
    
    A = [ skew(r1) * [eye(3) [0 0 0]']; skew(r2) * [R12 T12]; skew(r3) * [R13 T13]];
    
    [~, ~, V] = svd(A);

    lwp = V(:,end);
    lwp = lwp ./ lwp(end);

    LWP3 = [LWP3, lwp(1:3,1)];
end



SLWP3 = LWP3.* 0.475;

DLWP3 = [];

for i=1:size(SLWP3,2)
    lswp = abs(SLWP3(:,i)' * PARAMS.GT_PLANE_NORMAL_VECTOR' + PARAMS.GT_PLANE_DISPLACEMENT) / norm(PARAMS.GT_PLANE_NORMAL_VECTOR);
    DLWP3 = [DLWP3, lswp];
end

figure();
h1 = histogram(DLWP3);
h1.BinWidth = 0.005;
title('Distance Distribution to Ground Truth Plane (Linear 3 Views)');
xlabel('Distance to Plane');
ylabel('Frequency');














OC1 = [];
OC2 = [];
OC3 = [];



skew_T3 = [0, -T13(3), T13(2); T13(3), 0, -T13(1); -T13(2), T13(1), 0];

E3 = skew_T3 * R13;

ss = size(LWP3, 2);

for i = 1:ss
    trd = LWP3(:,i);

    r1 = rp(eye(3), [0 0 0]', K, trd);
    r2 = rp(R12, T12, K, trd);
    r3 = rp(R13, T13, K, trd);

    X = [r1(1,1), r1(2,1), r2(1,1), r2(2,1), r3(1,1), r3(2,1), 1, 10];

    Energy3 = @(x) (evaluateEnergyFunction3(x, c231(1:2,i)', c232(1:2,i)', c233(1:2,i)', E, E3));
    options = optimoptions('lsqnonlin', 'Display', 'iter');
    options.Algorithm = 'levenberg-marquardt';

    optimal_points = lsqnonlin(Energy3, X, [], [], options);


    oc1 = optimal_points(1:2);
    oc2 = optimal_points(3:4);
    oc3 = optimal_points(5:6);

    OC1 = cat(2, OC1, [oc1 1]');
    OC2 = cat(2, OC2, [oc2 1]');
    OC3 = cat(2, OC3, [oc3 1]');
end

KLWP3 = [];

for i=1:ss
    r1 = ik*OC1(:,i);
    r2 = ik*OC2(:,i);
    r3 = ik*OC3(:,i);

    A = [ skew(r1) * [eye(3) [0 0 0]']; skew(r2) * [R12 T12]; skew(r3) * [R13 T13]];

    [~, ~, V] = svd(A);

    klwp = V(:,end);
    klwp = klwp ./ klwp(end);

    KLWP3 = [KLWP3, klwp(1:3,1)];
end



SKLWP3 = KLWP3.* 0.475;

DKLWP3 = [];

for i=1:size(SKLWP3,2)
    lswp = abs(SKLWP3(:,i)' * PARAMS.GT_PLANE_NORMAL_VECTOR' + PARAMS.GT_PLANE_DISPLACEMENT) / norm(PARAMS.GT_PLANE_NORMAL_VECTOR);
    DKLWP3 = [DKLWP3, lswp];
end

figure();
h1 = histogram(DKLWP3);
h1.BinWidth = 0.005;
title('Distance Distribution to Ground Truth Plane (Kanatani 3 Views)');
xlabel('Distance to Plane');
ylabel('Frequency');













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

function Energy = evaluateEnergyFunction(x, gamma, gamma_bar, E)
    gamma_hat = x(1:2);
    gamma_bar_hat = x(3:4);
    lambda = x(5);

    term1 = norm(gamma_hat - gamma)^2;
    term2 = norm(gamma_bar_hat - gamma_bar)^2;

    term3 = lambda.*([gamma_bar_hat, 1]*E* [gamma_hat, 1]');

    Energy = term1 + term2 + term3;
end


function Energy3 = evaluateEnergyFunction3(x, gamma, gamma_bar, gamma_bar3, E, E2)
    gamma_hat = x(1:2);
    gamma_bar_hat = x(3:4);
    gamma_bar_hat3 = x(5:6);
    lambda = x(7);
    lambda2 = x(8);

    term1 = norm(gamma_hat - gamma)^2;
    term2 = norm(gamma_bar_hat - gamma_bar)^2;
    term3 = norm(gamma_bar_hat3 - gamma_bar3)^2;

    term4 = lambda.*([gamma_bar_hat, 1]*E* [gamma_hat, 1]');
    term5 = lambda2.*([gamma_bar_hat3, 1]*E2* [gamma_hat, 1]');

    Energy3 = term1 + term2 + term3 + term4 + term5;
end
