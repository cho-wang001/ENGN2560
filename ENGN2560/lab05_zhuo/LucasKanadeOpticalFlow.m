function [du, dv] = LucasKanadeOpticalFlow(Img1, Img2, Corners, Patch_Size)
	
%> Code Description: 
%     Given two images with initial corner locations on the first image, do
%     Lucas-Kanade optical flow algprothm which finds the displacements of
%     the corners from image 1 to 2 based on photometric consistency.
%
%> Inputs:                     
%     Img1:       Image 1 in gray scale with class type 'double'.  
%     Img2:       Image 2 with the same format as Image 1.
%     Corners:    A Nx2 array of corner locations where N is the total 
%                 number of corners detected on the first image. 
%     Patch_Size: The size of a patch (or window) attached to each corner
%
%> Outputs:
%     du:         A Nx1 array of displacements in x direction; each entry 
%                 corresponds to the entry of the Corners array.
%     dv:         A Nx1 array of displacements in y direction; each entry 
%                 corresponds to the entry of the Corners array.
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Feb. 22nd, 2024

    %> Pad the images
    w = floor(Patch_Size/2);
    Img1_Padded = padarray(Img1, [w+1 w+1], 'replicate');
    Img2_Padded = padarray(Img2, [w+1 w+1], 'replicate');
    
    %> Compute the image gradient
    [Ix, Iy] = gradient(Img2_Padded);
    It = Img1_Padded - Img2_Padded;

    %> Create griddedinterpolant structures for image gradient
    [X, Y] = meshgrid(1:size(Img1_Padded,2), 1:size(Img1_Padded,1));
    X = X'; Y = Y'; Ix = Ix'; Iy = Iy'; It = It';
    Ix_interp = griddedInterpolant(X, Y, Ix);
    Iy_interp = griddedInterpolant(X, Y, Iy);
    It_interp = griddedInterpolant(X, Y, It);

    %> Initialize the flow field for corners
    du = zeros(size(Corners,1), 1);
    dv = zeros(size(Corners,1), 1);
    
    %> Loop over all corners
    for ci = 1:size(Corners,1)

        %> Get corner locations
        cx = Corners(ci,1)+w+1;
        cy = Corners(ci,2)+w+1;

        %> Get the locations of the patch attached to the corner
        [Patch_X, Patch_Y] = meshgrid(cx-w:cx+w, cy-w:cy+w);
        Patch_X = Patch_X'; Patch_Y = Patch_Y';

        %> Do interpolation to get photometry of the patch
        Ix_Patch = Ix_interp(Patch_X, Patch_Y);
        Iy_Patch = Iy_interp(Patch_X, Patch_Y);
        It_Patch = It_interp(Patch_X, Patch_Y);
        Ix_Patch = Ix_Patch(:);
        Iy_Patch = Iy_Patch(:);
        It_Patch = It_Patch(:);

        %> Compute the flow from the equation of Lucas-Kanade algorithm
        A = [Ix_Patch, Iy_Patch];
        dP = inv(A'*A) * A' * It_Patch;
        du(ci) = dP(1);
        dv(ci) = dP(2);
    end
end
