%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab02: Feature Correspondences, Camera Relative Pose, and RANSAC
%> Problem4: Estimate Relative Pose Under a RANSAC Scheme
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Load two images
Img1 = imread('data/Problem2/1.png');
Img2 = imread('data/Problem2/2.png');

%> Load camera intrinsic matrix
load('data/Problem3/IntrinsicMatrix.mat');

%> Parameters passed to RANSAC
PARAMS.INLIER_THRESH     = 2;       %> 2 pixels
PARAMS.RANSAC_ITERATIONS = 2000;    %> Total number of RANSAC iterations

%> =================================================================
%> TODO: Estimate E from RANSAC and recover veridical (R,T) from E
%> =================================================================

load('data/Problem3/RelativeRotation.mat');
load('data/Problem3/RelativeTranslation.mat');

if size(Img1, 3) == 3
    Img1 = rgb2gray(Img1);
end

if size(Img2, 3) == 3
    Img2 = rgb2gray(Img2);
end

PeakThresh = 0.001;
EdgeThresh = 100;

[f1, d1] = vl_sift(single(Img1), 'PeakThresh', PeakThresh, 'EdgeThresh', EdgeThresh);
[f2, d2] = vl_sift(single(Img2), 'PeakThresh', PeakThresh, 'EdgeThresh', EdgeThresh);

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

[EE, II] = Ransac4Essential(PARAMS, mf1', mf2', K);

[U, S, V] = svd(EE);
W = [0, -1, 0; 1, 0, 0; 0, 0, 1];

R1 = U * W * V';
T1 = U(:, 3);
R2 = U * W' * V';
T2 = -U(:, 3);

if det(R1) < 0
    R1 = -R1;
    T1 = -T1;
end

if det(R2) < 0
    R2 = -R2;
    T2 = -T2;
end

s = size(mf1, 2);
invk = inv(K);

gmf1 = [];
gmf2 = [];

for i = 1:s
    rmf1 = invk*[mf1(:,i); 1];
    gmf1 = cat(2, gmf1, rmf1);
end

for i = 1:s
    rmf2 = invk*[mf2(:,i); 1];
    gmf2 = cat(2, gmf2, rmf2);
end

s1 = 0;
s2 = 0;
s3 = 0;
s4 = 0;

for i = 1:s
    A = [-R1 * gmf1(:,i), gmf2(:,i)];
    b = T1;
    [Ua, Sa, Va] = svd(A, 'econ');
    rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
    rho1 = rhos(1);
    rho2 = rhos(2);
    
    if rho1>0 && rho2>0
        s1 = s1+1;
    end

end

for i = 1:s
    A = [-R1 * gmf1(:,i), gmf2(:,i)];
    b = T2;
    [Ua, Sa, Va] = svd(A, 'econ');
    rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
    rho1 = rhos(1);
    rho2 = rhos(2);
    
    if rho1>0 && rho2>0
        s2 = s2+1;
    end

end

for i = 1:s
    A = [-R2 * gmf1(:,i), gmf2(:,i)];
    b = T1;
    [Ua, Sa, Va] = svd(A, 'econ');
    rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
    rho1 = rhos(1);
    rho2 = rhos(2);
    
    if rho1>0 && rho2>0
        s3 = s3+1;
    end

end

for i = 1:s
    A = [-R2 * gmf1(:,i), gmf2(:,i)];
    b = T2;
    [Ua, Sa, Va] = svd(A, 'econ');
    rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
    rho1 = rhos(1);
    rho2 = rhos(2);
    
    if rho1>0 && rho2>0
        s4 = s4+1;
    end

end

SS = [s1 s2 s3 s4];

[~, index] = max(SS);

if index==1
    disp('Veridical R and T are R1 and T1');
    RR = R1;
    TT = T1;
elseif index==2
    disp('Veridical R and T are R1 and T2');
    RR = R1;
    TT = T2;
elseif index==3
    disp('Veridical R and T are R2 and T1');
    RR = R2;
    TT = T1;
else
    disp('Veridical R and T are R2 and T2');
    RR = R2;
    TT = T2;
end

rg = R;
tg = T;

rotation_error = real(acos(0.5 * (trace(rg * RR' - eye(3)))));
translation_error = abs(dot(tg, TT) - 1);
disp("rotation_error:");
disp(rotation_error);
disp("translation_error:");
disp(translation_error);

function [E, inlier_Idx] = Ransac4Essential(PARAMS,gamma1,gamma2,K)
    I = PARAMS.RANSAC_ITERATIONS;
    T = PARAMS.INLIER_THRESH;

    E = [];

    ml = 0;

    K = inv(K);

    for i = 1:I
        c1m = [];
        c2m = [];
        nm = size(gamma1, 1);
        sc = randsample(nm, 5,'true');

        for ii=sc
            x_coord1 = gamma1(ii,1);
            y_coord1 = gamma1(ii,2);
            x_coord2 = gamma2(ii,1);
            y_coord2 = gamma2(ii,2);
        end

        cc1 = [x_coord1, y_coord1];
        cc2 = [x_coord2, y_coord2];
        
        for iii = 1:size(cc1, 1)
            g1 = K*[cc1(iii, 1);cc1(iii, 2);1];
            g1 = g1';
            c1m = [c1m; g1];
        
            g2 = K*[cc2(iii, 1);cc2(iii, 2);1];
            g2 = g2';
            c2m = [c2m; g2];
        end
        
        mi = cat(3, c1m, c2m);
        
        es = fivePointAlgorithmSelf(mi);
        
        for j = 1:size(es, 3)
            e = es{:,:,j};
            
            l = 0;

            for n = 1:size(gamma1, 1)
                c13d = K*[gamma1(n, 1); gamma1(n, 2); 1];

                A = c13d(1, 1)*e(1,1) + c13d(2, 1)*e(1,2) + e(1,3);
                B = c13d(1, 1)*e(2,1) + c13d(2, 1)*e(2,2) + e(2,3);
                C = c13d(1, 1)*e(3,1) + c13d(2, 1)*e(3,2) + e(3,3);
    
                aa = A;
                bb = B;
                cc = C;

                A = aa * K(1,1);
                B = bb * K(2,2);
                C = aa * K(1,3) + bb * K(2,3) + cc;

                distance = abs(A * gamma2(n, 1) + B * gamma2(n, 2) + C) / sqrt(A^2 + B^2);

                if distance<T
                    l = l+1;
                end

            end

            if l >= ml
                ml = l;
                E = e;
            end

        end
        
    end
    

    inlier_Idx = ml;

end