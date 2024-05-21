function [Points2D_Location, Points2D_Grad_Mag] = get_Candidate_Points(PARAMS, Img)

%> Code Description: 
%     Given an image, pick randomly and uniformly distributed 2D points that 
%     are "distinct" enough for a direct sparse visual odometry.
%
%> Inputs: 
%     PARAMS:            A parameter structure.
%     Img:               An image of class 'double'.
%
%> Outputs:
%     Points2D_Location: A 2xN matrix of column-wise structured 2D point locations
%                        picked from the input image. 
%     Points2D_Grad_Mag: A 1xN vector of the corresponding gradient magnitude of the 
%                        2D points.
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Mar. 10th, 2024

    Img_Width  = size(Img, 2);
    Img_Height = size(Img, 1);
    X = unifrnd(0, Img_Width,  1, PARAMS.NUM_OF_UNIFORMLY_DISTRIBUTED_POINTS);
    Y = unifrnd(0, Img_Height, 1, PARAMS.NUM_OF_UNIFORMLY_DISTRIBUTED_POINTS);
    Points2D = [X; Y];

    %> Compute the image gradient of Points2D
    [X, Y] = meshgrid(1:size(Img,2), 1:size(Img,1));
    [Ix, Iy] = gradient(Img);
    X = X'; Y = Y'; Ix = Ix'; Iy = Iy';
    Ix_interp = griddedInterpolant(X, Y, Ix);
    Iy_interp = griddedInterpolant(X, Y, Iy);

    Num_Of_Blocks_in_X = size(Img,2) / PARAMS.POINTS_BLOCK_SIZE;
    Num_Of_Blocks_in_Y = size(Img,1) / PARAMS.POINTS_BLOCK_SIZE;
    Index_Block_in_X   = cell(Num_Of_Blocks_in_X, 1);
    Index_Block_in_Y   = cell(Num_Of_Blocks_in_Y, 1);
    % Points2D_Candidate = cell(Num_Of_Blocks_in_Y, Num_Of_Blocks_in_X);

    %> X (column) direction
    counter_X = 1;
    for ci = 1:PARAMS.POINTS_BLOCK_SIZE:size(Img,2)
        Index_Block_in_X{counter_X} = find(Points2D(1,:) < ci+PARAMS.POINTS_BLOCK_SIZE & Points2D(1,:) >= ci);
        counter_X = counter_X + 1;
    end

    %> Y (row) direction
    counter_Y = 1;
    for ri = 1:PARAMS.POINTS_BLOCK_SIZE:size(Img,1)
        Index_Block_in_Y{counter_Y} = find(Points2D(2,:) < ri+PARAMS.POINTS_BLOCK_SIZE & Points2D(2,:) >= ri);
        counter_Y = counter_Y + 1;
    end

    %> Assertion check
    assert(counter_X == Num_Of_Blocks_in_X+1);
    assert(counter_Y == Num_Of_Blocks_in_Y+1);

    %> Loop over all blocks and store 2D points per block
    Points2D_Location = [];
    Points2D_Grad_Mag = [];
    for ri = 1:Num_Of_Blocks_in_Y
        for ci = 1:Num_Of_Blocks_in_X
            [~, ix, ~] = intersect(Index_Block_in_X{ci,1}, Index_Block_in_Y{ri,1});
            Points2D_in_Block = Points2D(:,Index_Block_in_X{ci,1}(ix));
            Ix_Points2D_in_Block = Ix_interp(Points2D_in_Block(1,:), Points2D_in_Block(2,:));
            Iy_Points2D_in_Block = Iy_interp(Points2D_in_Block(1,:), Points2D_in_Block(2,:));
            gradI_mag = vecnorm([Ix_Points2D_in_Block; Iy_Points2D_in_Block], 2, 1);
            Candidate_Index = gradI_mag > (median(gradI_mag) + PARAMS.GRAD_THRESH_FOR_CANDIDATE_POINTS);

            Points2D_Location = [Points2D_Location, Points2D_in_Block(:,Candidate_Index)];
            Points2D_Grad_Mag = [Points2D_Grad_Mag, gradI_mag(:,Candidate_Index)];
        end
    end
end
