function draw_Feature_Matches(im1, im2, matched_f1, matched_f2, N, disp_Img_SideBySide)

%> Code Description: 
%     Given two images and their feature locations, visualize the feature
%     correspondences.
%
%> Inputs: 
%     im1:                  Image 1. Suggested to be a gray image as feature
%                           correspondences can be popped out clearly.
%     im2:                  Image 2. Also suggested to be a gray image.
%     matched_f1:           Matched feature locations in image 1, structured 
%                           in a 2xM matrix where M is the total number of 
%                           feature correspondences.
%     matched_f2:           Matched feature locations in image 2, structured 
%                           the same way as matched_f1.
%     N:                    The top N rank-ordered list to show feature
%                           correspondences, e.g. N=30 means showing only 
%                           the top 30 matches from the rank-ordered list.
%     disp_Img_SideBySide:  A binary number deciding whether the image pair
%                           is shown side by side (if disp_Img_SideBySide=1) 
%                           or up and down (if disp_Img_SideBySide=0).
%
%> Outputs:
%     None
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Dec. 12th, 2023

    figure; clf;
    if disp_Img_SideBySide == 1
        imshow([im1, im2]);
    else
        imshow([im1; im2]);
    end
    hold on;
    axis image off;

    xa = matched_f1(1,1:N);
    xb = matched_f2(1,1:N);
    ya = matched_f1(2,1:N);
    yb = matched_f2(2,1:N);

    disp_RGB_colors = 0.5 + 0.5 .* (2*rand([N, 3], 'like', 0.5)-1);

    %> 1) Points
    if disp_Img_SideBySide == 1
        for i = 1:N
            plot(xa(i), ya(i), 's', 'Color', disp_RGB_colors(i,:), 'LineWidth', 2);
            hold on;
            plot(xb(i) + size(im1,2), yb(i), 's', 'Color', disp_RGB_colors(i,:), 'LineWidth', 2);
            hold on;
            h = line([xa(i) ; xb(i) + size(im1,2)], ...
                      [ya(i) ; yb(i)]);
            hold on;
            set(h, 'linewidth', 1, 'color', disp_RGB_colors(i,:));
        end
        hold off;
        xlim([1, size(im1,2) + size(im1,2)]);
        ylim([1, size(im1,1)]);
    else
        for i = 1:N
            plot(xa(i), ya(i), 's', 'Color', disp_RGB_colors(i,:), 'LineWidth', 2);
            hold on;
            plot(xb(i), yb(i) + size(im1,1),  's', 'Color', disp_RGB_colors(i,:), 'LineWidth', 2);
            hold on;
            h = line([xa(i) ; xb(i) ], ...
                      [ya(i) ; yb(i) + size(im1,1)]);
            hold on;
            set(h,'linewidth', 1, 'color', 'r');
        end
        hold off;
        xlim([1, size(im1,2)]);
        ylim([1, size(im1,1) + size(im1,1)]);
    end
end