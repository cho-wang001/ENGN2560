%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab01: Camera Projection, Camera Calibration, and Image Features
%> Problem1: Image Formation by Camera Projection
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Load the colored point cloud
load('data/Problem1/scenePointCloud.mat');

%> Load the camera poses (rotation and translation)
load('data/Problem1/cameraPoses.mat');

%> Load the camera calibration matrix
load('data/Problem1/cameraIntrinsicMatrix.mat');

%> Show a cloud of 3D points and the cameras
axes1 = pcshow(scenePointCloud); 
hold on;
plotCamera('Location',-R1'*T1,'Orientation',R1,'Opacity',0,'Size',20);
plotCamera('Location',-R2'*T2,'Orientation',R2,'Opacity',0,'Size',20); 
grid(axes1,'on');
axis(axes1,'tight');
set(axes1,'CameraPosition',...
    [5264.74608400312 -1534.61734696361 128.670524332434],'CameraUpVector',...
    [-0.0833474512990768 0.0254818730835091 -0.996194698091746],...
    'DataAspectRatio',[1 1 1]);
hold off;

disp(scenePointCloud);
set(gca,'color','w');
set(gcf,'color','w');

%> ===========================================
%> TODO: Project 3D points onto image planes
%> ===========================================

wc = scenePointCloud.Location;

num = size(wc, 1);

ic1 = zeros(num, 2);
ic2 = zeros(num, 2);

for i = 1:num
    xw = wc(i, 1);
    yw = wc(i, 2);
    zw = wc(i, 3);

    v = [xw, yw, zw];
    v = v';

    o1 = R1*v + T1;
    o2 = R2*v + T2;

    o1 = K*o1;
    o2 = K*o2;

    o1 = o1./o1(3);
    o2 = o2./o2(3);

    if (o1(1) >= 1 && o1(1) <= 1600) && (o1(2) >= 1 && o1(2) <= 1200)
       ic1(i, 1) = o1(2);
       ic1(i, 2) = o1(1);
    end

    if (o2(1) >= 1 && o2(1) <= 1600) && (o2(2) >= 1 && o2(2) <= 1200)
       ic2(i, 1) = o2(2);
       ic2(i, 2) = o2(1);
    end

end

colors = scenePointCloud.Color;

imc1 = zeros(1200, 1600, 3);
imc2 = zeros(1200, 1600, 3);

for j = 1:num
    pic1 = ic1(j,:);
    pic1x = pic1(1);
    pic1y = pic1(2);

    pic2 = ic2(j,:);
    pic2x = pic2(1);
    pic2y = pic2(2);

    if pic1x > 0
        r1 = colors(j, 1);
        g1 = colors(j, 2);
        b1 = colors(j, 3);

        pic1x = round(pic1x);
        pic1y = round(pic1y);

        imc1(pic1x, pic1y, 1) = r1;
        imc1(pic1x, pic1y, 2) = g1;
        imc1(pic1x, pic1y, 3) = b1;
    end

    if pic2x > 0
        r2 = colors(j, 1);
        g2 = colors(j, 2);
        b2 = colors(j, 3);

        pic2x = round(pic2x);
        pic2y = round(pic2y);

        imc2(pic2x, pic2y, 1) = r2;
        imc2(pic2x, pic2y, 2) = g2;
        imc2(pic2x, pic2y, 3) = b2;
    end
end

figure();
imshow(uint8(imc1));

figure();
imshow(uint8(imc2));

[xm, ym] = meshgrid(1:1600, 1:1200);

x1 = ic1(:, 2);
y1 = ic1(:, 1);

r = double(colors(:, 1));
g = double(colors(:, 2));
b = double(colors(:, 3));

p1 = zeros(1200, 1600, 3);
p1(:, :, 1) = griddata(x1, y1, r, xm, ym);
p1(:, :, 2) = griddata(x1, y1, g, xm, ym);
p1(:, :, 3) = griddata(x1, y1, b, xm, ym);

figure();
imshow(uint8(p1));

x2 = ic2(:, 2);
y2 = ic2(:, 1);

r = double(colors(:, 1));
g = double(colors(:, 2));
b = double(colors(:, 3));

p2 = zeros(1200, 1600, 3);
p2(:, :, 1) = griddata(x2, y2, r, xm, ym);
p2(:, :, 2) = griddata(x2, y2, g, xm, ym);
p2(:, :, 3) = griddata(x2, y2, b, xm, ym);

figure();
imshow(uint8(p2));

