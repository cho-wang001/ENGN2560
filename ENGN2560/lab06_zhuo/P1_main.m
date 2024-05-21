%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab06: Direct Sparse Visual Odometry
%> ------------------------------------------------------------------------
clc; clear all; close all;
% rng(0);

%> Read all undistorted images in the sequence.
%> Use imread(Image_Sequence(i).name); to read image i
mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir([mfiledir, '/data/seq_07/*.png']);

%> Load intrinsic matrix
load("data/IntrinsicMatrix.mat");

%> Load ground-truth poses
load("data/GT_Poses.mat");

PARAMS.NUM_OF_UNIFORMLY_DISTRIBUTED_POINTS  = 2000; %> Same as DSO
PARAMS.POINTS_BLOCK_SIZE                    = 32;   %> Same as in DSO
PARAMS.GRAD_THRESH_FOR_CANDIDATE_POINTS     = 3.0;  %> Made differently from DSO
PARAMS.PYRAMID_TOTAL_LEVELS                 = 3;
PARAMS.PATCH_SIZE                           = 5;
PARAMS.SSD_THRESH                           = 750;
PARAMS.HUBER_THRESH                         = 9;    %> Same as DSO
PARAMS.GRAD_DEPENDENT_WEIGHT                = 50;   %> Same as DSO

%> Some initial guesses
PARAMS.INVERSE_DEPTH_SAMPLES                = 0.2:0.05:2; %> Inverse Depth Samples
PARAMS.BRIGHTNESS_TRANSFER_PARAM_A          = 1;
PARAMS.BRIGHTNESS_TRANSFER_PARAM_B          = 0;

%> ========================================================================
%> TODO: Do a simple direct visual odometry from a sequence of 
%        photometrically and geometrically calibrated images 
%> ========================================================================

s = size(Image_Sequence,1);

img = [];

for i = 1:s
    f = Image_Sequence(i).folder;
    n = Image_Sequence(i).name;

    p = [f '\' n];

    image = imread(p);

    if size(image, 3) == 3
        image = double(rgb2gray(image));
    else
        image = double(image);
    end
    

    img = cat(3, img, image);
end



image_pyramid = cell(1, PARAMS.PYRAMID_TOTAL_LEVELS);

image_pyramid{1} = img(:,:,1);


for i = 2:PARAMS.PYRAMID_TOTAL_LEVELS
    image_pyramid{i} = impyramid(image_pyramid{i-1}, 'reduce');
end

k = [];

for l = 1:PARAMS.PYRAMID_TOTAL_LEVELS

    kk = [0.5^(l-1)*K(1,1), 0, 0.5^(l-1)*(K(1,3)+0.5)-0.5;
      0, 0.5^(l-1)*K(2,2), 0.5^(l-1)*(K(2,3)+0.5)-0.5;
      0, 0, 1];

    k = cat(3, k, kk);
end

[lo, mn] = get_Candidate_Points(PARAMS, img(:,:,1));

sf = 0.5^(numel(image_pyramid)-1);

lot = lo*sf;
lot = lot';

O_Poses = zeros(3,4,7);

O_Poses(:,:,1) = [1,0,0,0;
                   0,1,0,0;
                   0,0,1,0];


for i=2:s
    if i == 2
        image_pyramidl = cell(1, PARAMS.PYRAMID_TOTAL_LEVELS);
        image_pyramidl{1} = img(:,:,i);
    else
        [lo, mn] = get_Candidate_Points(PARAMS, img(:,:,i-1));
        lot = lo*sf;
        lot = lot';

        image_pyramid = image_pyramidl;
        image_pyramidl = cell(1, PARAMS.PYRAMID_TOTAL_LEVELS);
        image_pyramidl{1} = img(:,:,i);

    end
    

    for j = 2:PARAMS.PYRAMID_TOTAL_LEVELS
        image_pyramidl{j} = impyramid(image_pyramidl{j-1}, 'reduce');
    end

    if i>2
        R = (GT_Poses(:,1:3,(i-1))*GT_Poses(:,1:3,(i-2))')*GT_Poses(:,1:3,(i-1));
        T = GT_Poses(:,4,(i-1)) + (GT_Poses(:,1:3,(i-1))*GT_Poses(:,1:3,(i-2))')*(GT_Poses(:,4,(i-1)) - GT_Poses(:,4,(i-2)));
        R_AxAng = rotm2axang(R);
        R_AxAng = R_AxAng(end).*R_AxAng(1,1:3);
    else
        R = GT_Poses(:,1:3,i);
        T = GT_Poses(:,4,i);
        R_AxAng = rotm2axang(R);
        R_AxAng = R_AxAng(end).*R_AxAng(1,1:3);
    end
    
    
    w = floor(PARAMS.PATCH_SIZE/2);
    Img1_Padded = padarray(double(image_pyramid{3}), [w+1 w+1], 'replicate');
    Img2_Padded = padarray(double(image_pyramidl{3}), [w+1 w+1], 'replicate');
    
    [X, Y] = meshgrid(1:size(Img1_Padded,2), 1:size(Img1_Padded,1));
    X = X'; Y = Y'; Img1_Padded = Img1_Padded'; Img2_Padded = Img2_Padded';
    Ix_interp = griddedInterpolant(X, Y, Img1_Padded);
    Iy_interp = griddedInterpolant(X, Y, Img2_Padded);
    
    d = size(lot,1);
    
    kkk = k(:,:,3);

    invk = inv(kkk);
    
    rc = [];
    b=[];
    id=[];
    
    for n = 1:d
        cx = lot(n,1)+w+1;
        cy = lot(n,2)+w+1;
    
        [Patch_X, Patch_Y] = meshgrid(cx-w:cx+w, cy-w:cy+w);
        Patch_X = Patch_X'; Patch_Y = Patch_Y';
    
        Ix_Patch = Ix_interp(Patch_X, Patch_Y);
        Ix_Patch = Ix_Patch(:);
    
        A = [Ix_Patch];

        pc = [lot(n,1), lot(n,2), 1]';

        mc = invk*pc;

        sd = [];
        sdd = [];
        cc2 = [];

        for p = PARAMS.INVERSE_DEPTH_SAMPLES
            wmc = mc.*(1/p);
    
            pc2 = kkk*(R*wmc + T);
    
            pc2 = pc2./pc2(end);
    
            pp = isPointOutsideImage(pc2, [120, 160]);

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

            sdds = (1./PARAMS.PATCH_SIZE)^2 .* sum(sum((A - B).^2));
            sdd = [sdd, sdds];

            [~, indexp] = min(sdd);
        end

        if size(sd) == 0
            b = [b, n];
            continue;
        end

        [~, index] = min(sd);
        
        pd = [PARAMS.INVERSE_DEPTH_SAMPLES];

        id = [id,pd(indexp)];

        rc = [rc, cc2(1:2,index)];

    end
    
    mf1 = lot;
    mf1(b,:) = [];
    mmn = mn;
    mmn(b) = [];
    can = ones(1, size(mf1,1));
    mf1 = cat(2, mf1, can');
    
    mf1 = mf1';

    mf1 = mf1.*id;

    
    v = zeros(1,8);

    v(1:3) = T';
    v(4:6) = R_AxAng;
    v(7) = PARAMS.BRIGHTNESS_TRANSFER_PARAM_A;
    v(8) = PARAMS.BRIGHTNESS_TRANSFER_PARAM_B;

    hh = reshape(mf1, 1, []);

    v = cat(2,v,hh);

    e = get_Photometric_Error(v, PARAMS,kkk, image_pyramid, image_pyramidl, mmn);

    ff = @(x)(get_Photometric_Error( v, PARAMS,kkk, image_pyramid, image_pyramidl, mmn));
    options = optimoptions('lsqnonlin', 'Display', 'iter');
    options.Algorithm = 'levenberg-marquardt';
    BA_Output = lsqnonlin(ff, v, [], [], options);
    
    ot = v(1:3)';
    ora = v(4:6);

    orr = norm(ora);
    or = axang2rotm([ora.*orr, orr]);

    ceng = [or,ot];

    O_Poses(:,:,i) = ceng;
end

Visualize_Trajectory(GT_Poses, O_Poses)





RMSER = 0;

for i = 1:s
    r = O_Poses(:,1:3,i);
    rg = GT_Poses(:, 1:3, i);
    re = (acos(0.5 * (trace(rg' * r) - 1)))^2;

    RMSER = RMSER + re;

end

RMSER = sqrt(RMSER/s);

RMSET = 0;

for i = 2:s
    t = O_Poses(:,4,i);
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











function Error = get_Photometric_Error(data, PARAMS,k, image_pyramid, image_pyramidl, mmn)

    t = data(1:3)';
    r = data(4:6);
    a = data(7);
    b = data(8);

    rr = norm(r);
    R = axang2rotm([r.*rr, rr]);

    

    x = data(9:end);
    GG = reshape(x, 3, size(x,2)/3);

    ro = GG(3,:);
    re = GG./ro;

    re = re(1:2,:);

    rmf1 = Rotate_Points_by_Axis_Angle_Rotation(r, GG);

    s = size(rmf1,2);
    
    ri = [];

    for i = 1:s
        g = rmf1(:,i);

        rc = rep(R,t,k,g);
        
        ri = [ri, rc];
    end
    

    ri = ri(1:2,:);

    
    w = floor(PARAMS.PATCH_SIZE/2);
    Img1_Padded = padarray(double(image_pyramid{3}), [w+1 w+1], 'replicate');
    Img2_Padded = padarray(double(image_pyramidl{3}), [w+1 w+1], 'replicate');
    
    [X, Y] = meshgrid(1:size(Img1_Padded,2), 1:size(Img1_Padded,1));
    X = X'; Y = Y'; Img1_Padded = Img1_Padded'; Img2_Padded = Img2_Padded';
    Ix_interp = griddedInterpolant(X, Y, Img1_Padded);
    Iy_interp = griddedInterpolant(X, Y, Img2_Padded);
    
    d = size(re,2);
    

    
    err = 0;
    
    for j = 1:d
        cx = re(1,j)+w+1;
        cy = re(2,j)+w+1;
    
        [Patch_X, Patch_Y] = meshgrid(cx-w:cx+w, cy-w:cy+w);
        Patch_X = Patch_X'; Patch_Y = Patch_Y';
    
        Ix_Patch = Ix_interp(Patch_X, Patch_Y);
        Ix_Patch = Ix_Patch(:);
    
        A = [Ix_Patch];

        cx2 = ri(1,j)+w+1;
        cy2 = ri(2,j)+w+1;
        
        [Patch_X2, Patch_Y2] = meshgrid(cx2-w:cx2+w, cy2-w:cy2+w);
        Patch_X2 = Patch_X2'; Patch_Y2 = Patch_Y2';
    
        Iy_Patch2 = Iy_interp(Patch_X2, Patch_Y2);
        Iy_Patch2 = Iy_Patch2(:);

        B = [Iy_Patch2];

        
        en = norm(B) - a.*norm(A) - b;
        
        if en<= PARAMS.HUBER_THRESH
            en = 0.5*(en)^2;
        else
            en = PARAMS.HUBER_THRESH*(abs(en)-0.5*(PARAMS.HUBER_THRESH));
        end

        wh = (en^2) / (en^2 + norm(mmn(j))^2); 
        
        err = err+wh;

    end

  Error = err;
end






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




function r = rep(R, T, K, g)
    ga = R*g +T;
    c = K*ga;
    c = c./c(end);

    r = c;
end

