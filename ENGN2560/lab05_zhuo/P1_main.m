4%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab05: Correspondences from Stereo Vision
%> Problem1: Multi-Resolution Optical Flow
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Read all images in the sequence.
%> Use imread(Image_Sequence(i).name); to read image i
mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir([mfiledir, '/data/Problem1/fr3_household/*.png']);

PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS  = 2;
PARAMS.LK_OPTICAL_FLOW_WINDOW_SIZE    = 21; 

%> =============================================================
%> TODO: Implement Multi-Resolution Lucas-Kanade Optical Flow
%> =============================================================

s = size(Image_Sequence,1);

img = [];

for i = 1:s
    f = Image_Sequence(i).folder;
    n = Image_Sequence(i).name;

    p = [f '\' n];

    image = imread(p);

    if size(image, 3) == 3
        image = im2double(rgb2gray(image));
    end
    

    img = cat(3, img, image);
end



image_pyramid = cell(1, PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS);

image_pyramid{1} = img(:,:,1);


for i = 2:PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS
    image_pyramid{i} = impyramid(image_pyramid{i-1}, 'reduce');
end

corners = detectHarrisFeatures(image_pyramid{1});

imshow(img(:,:,1));
hold on;
plot(corners);
hold on;

for i = 2:numel(image_pyramid)
    scale_factor = 0.5^(numel(image_pyramid) - i + 1);

    corners.Location = corners.Location * scale_factor;


    if i < numel(image_pyramid)
        image_pyramid{i+1} = impyramid(image_pyramid{i}, 'reduce');
        corners = detectHarrisFeatures(image_pyramid{i+1});
    end
end




for i =2:s

    image_pyramidl = cell(1, PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS);

    image_pyramidl{1} = img(:,:,i);


    for j = 2:PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS
        image_pyramidl{j} = impyramid(image_pyramidl{j-1}, 'reduce');
    end

    [du, dv] = LucasKanadeOpticalFlow(image_pyramid{PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS}, image_pyramidl{PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS}, corners.Location, PARAMS.LK_OPTICAL_FLOW_WINDOW_SIZE);

    cornersl(:,1) = corners.Location(:,1) + du;
    cornersl(:,2) = corners.Location(:,2) + dv;

    cornersl = cornersl.*2.*(PARAMS.LK_OPTICAL_FLOW_NUM_OF_LEVELS - 1);

    cornersl = removeOutOfBoundPoints(cornersl, [480, 640]);

    scatter(cornersl(:,1), cornersl(:,2),'+');
    hold on;

    corners.Location = cornersl * scale_factor;


end

hold off;

function valid_points = removeOutOfBoundPoints(points, image_size)
    x = points(:, 1);
    y = points(:, 2);

    valid_indices = x >= 1 & x <= image_size(2) & y >= 1 & y <= image_size(1);

    valid_points = points(valid_indices, :);
end

