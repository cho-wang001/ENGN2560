%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab05: Correspondences from Stereo Vision
%> Problem3: Stereo Matching Based on Epipolar Geometry
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Read an image pair
Img1 = imread('data/Problem2/Img1.png');
Img2 = imread('data/Problem2/Img2.png');

%> Load camera intrinsic matrix
load('data/Problem2/IntrinsicMatrix.mat');

%> Known (ground-truth) relative pose
load('data/Problem2/GT_R.mat');		%> R_gt
load('data/Problem2/GT_T.mat');		%> T_gt

PARAMS.POINT_TO_EPIPOLAR_LINE_DIST      = 5;    %> 5 pixels
PARAMS.PATCH_SIZE                       = 7;    %> 7x7 window
PARAMS.SSD_THRESH                       = 900;	%> Threshold for average SSD  

%> =======================================================================
%> TODO: Find corner matches from a stereo images using epipolar geometry
%> =======================================================================



E = skew(T_gt) * R_gt;
f = inv(K)' * E * inv(K);



if size(Img1, 3) == 3
    Img1 = double(rgb2gray(Img1));
end

if size(Img2, 3) == 3
    Img2 = double(rgb2gray(Img2));
end

corners1 = detectHarrisFeatures(Img1);
corners2 = detectHarrisFeatures(Img2);


figure();
imshow(uint8(Img1)); hold on;
plot(corners1);

w = floor(PARAMS.PATCH_SIZE/2);
Img1_Padded = padarray(Img1, [w+1 w+1], 'replicate');
Img2_Padded = padarray(Img2, [w+1 w+1], 'replicate');

[X, Y] = meshgrid(1:size(Img1_Padded,2), 1:size(Img1_Padded,1));
%[Ix, Iy] = gradient(Img2_Padded);
%It = Img1_Padded - Img2_Padded;
X = X'; Y = Y'; Img1_Padded = Img1_Padded'; Img2_Padded = Img2_Padded'; %It = It';
Ix_interp = griddedInterpolant(X, Y, Img1_Padded);
Iy_interp = griddedInterpolant(X, Y, Img2_Padded);
%It_interp = griddedInterpolant(X, Y, It);

s = size(corners1.Location,1);
ss = size(corners2.Location,1);

rc = [];
b = [];

for i = 1:s
    cx = corners1.Location(i,1)+w+1;
    cy = corners1.Location(i,2)+w+1;

    [Patch_X, Patch_Y] = meshgrid(cx-w:cx+w, cy-w:cy+w);
    Patch_X = Patch_X'; Patch_Y = Patch_Y';

    Ix_Patch = Ix_interp(Patch_X, Patch_Y);
    %Iy_Patch = Iy_interp(Patch_X, Patch_Y);
    Ix_Patch = Ix_Patch(:);
    %Iy_Patch = Iy_Patch(:);

    A = [Ix_Patch];
    

    sd = [];
    cc2 = [];

    for p = 1:ss

        pc2 = [corners2.Location(p,1); corners2.Location(p,2)];

        c13d = [corners1.Location(i,1); corners1.Location(i,2); 1];
    
        AA = c13d(1, 1)*f(1,1) + c13d(2, 1)*f(1,2) + f(1,3);
        BB = c13d(1, 1)*f(2,1) + c13d(2, 1)*f(2,2) + f(2,3);
        C = c13d(1, 1)*f(3,1) + c13d(2, 1)*f(3,2) + f(3,3);

        distance = abs(AA * corners2.Location(p,1) + BB * corners2.Location(p,2) + C) / sqrt(AA^2 + BB^2);

        if distance > PARAMS.POINT_TO_EPIPOLAR_LINE_DIST
            continue;
        end
        
        cx2 = corners2.Location(p,1)+w+1;
        cy2 = corners2.Location(p,2)+w+1;
    
        [Patch_X2, Patch_Y2] = meshgrid(cx2-w:cx2+w, cy2-w:cy2+w);
        Patch_X2 = Patch_X2'; Patch_Y2 = Patch_Y2';
    
        %Ix_Patch2 = Ix_interp(Patch_X2, Patch_Y2);
        Iy_Patch2 = Iy_interp(Patch_X2, Patch_Y2);
        %Ix_Patch2 = Ix_Patch2(:);
        Iy_Patch2 = Iy_Patch2(:);


        B = [Iy_Patch2];

        ssd = (1./PARAMS.PATCH_SIZE)^2 .* sum(sum((A - B).^2));

        if ssd<PARAMS.SSD_THRESH
            sd = [sd, ssd];
            cc2 = [cc2, pc2];
        end
    end
    
    if isempty(sd)
        b = [b, i];
        continue;
    end

    [~, index] = min(sd);
    
    rc = [rc, cc2(1:2,index)];
end

mf1 = double(corners1.Location);
mf1(b,:) = [];

rc = double(rc);



draw_Feature_Matches(uint8(Img1), uint8(Img2), mf1', rc, 200, 1);





function S = skew(v)

    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end