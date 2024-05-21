%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab05: Correspondences from Stereo Vision
%> Problem2: Stereo Matching Based on Inverse Depths
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

%> Parameters
PARAMS.PATCH_SIZE               = 13;            %> 13x13 window
PARAMS.SSD_THRESH               = 900;           %> Threshold for average SSD         
PARAMS.DEPTH_SAMPLES            = 0.2:0.05:2.1;  %> Inverse depth samples

%> =======================================================================
%> TODO: Find potential matches from a stereo images using inverse depths
%> =======================================================================

if size(Img1, 3) == 3
    Img1 = rgb2gray(Img1);
end

if size(Img2, 3) == 3
    Img2 = rgb2gray(Img2);
end

corners1 = detectHarrisFeatures(Img1);

figure();
imshow(Img1); hold on;
plot(corners1);

w = floor(PARAMS.PATCH_SIZE/2);
Img1_Padded = padarray(double(Img1), [w+1 w+1], 'replicate');
Img2_Padded = padarray(double(Img2), [w+1 w+1], 'replicate');

[X, Y] = meshgrid(1:size(Img1_Padded,2), 1:size(Img1_Padded,1));
X = X'; Y = Y'; Img1_Padded = Img1_Padded'; Img2_Padded = Img2_Padded';
Ix_interp = griddedInterpolant(X, Y, Img1_Padded);
Iy_interp = griddedInterpolant(X, Y, Img2_Padded);

s = size(corners1.Location,1);

invk = inv(K);

rc = [];
b=[];

for i = 1:s
    cx = corners1.Location(i,1)+w+1;
    cy = corners1.Location(i,2)+w+1;

    [Patch_X, Patch_Y] = meshgrid(cx-w:cx+w, cy-w:cy+w);
    Patch_X = Patch_X'; Patch_Y = Patch_Y';

    Ix_Patch = Ix_interp(Patch_X, Patch_Y);
    Ix_Patch = Ix_Patch(:);

    A = [Ix_Patch];
    
    pc = [corners1.Location(i,1), corners1.Location(i,2), 1]';
    
    mc = invk*pc;

    sd = [];
    cc2 = [];

    for p = PARAMS.DEPTH_SAMPLES
        wmc = mc.*(1/p);

        pc2 = K*(R_gt*wmc + T_gt);

        pc2 = pc2./pc2(end);

        pp = isPointOutsideImage(pc2, [480, 640]);

        if pp
            continue;
        end

        cx2 = pc2(1)+w+1;
        cy2 = pc2(2)+w+1;
    
        [Patch_X2, Patch_Y2] = meshgrid(cx2-w:cx2+w, cy2-w:cy2+w);
        Patch_X2 = Patch_X2'; Patch_Y2 = Patch_Y2';
    
        Iy_Patch2 = Iy_interp(Patch_X2, Patch_Y2);
        Iy_Patch2 = Iy_Patch2(:);

        B = [Iy_Patch2];

        ssd = (1./PARAMS.PATCH_SIZE)^2 .* sum(sum((A - B).^2));

        if ssd<PARAMS.SSD_THRESH
            sd = [sd, ssd];
            cc2 = [cc2, pc2];
        end
    end
    
    if size(sd) == 0
        b = [b, i];
        continue;
    end

    [~, index] = min(sd);
    
    rc = [rc, cc2(1:2,index)];
end

mf1 = double(corners1.Location);
mf1(b,:) = [];

rc = double(rc);

draw_Feature_Matches(Img1, Img2, mf1', rc, 100, 1);


function outside = isPointOutsideImage(point, imageSize)

    rows = imageSize(1);
    cols = imageSize(2);

    x = point(1);
    y = point(2);

    if (x >= 1) && (x <= cols) && (y >= 1) && (y <= rows)
        outside = 0; 
    else
        outside = 1;
    end
end