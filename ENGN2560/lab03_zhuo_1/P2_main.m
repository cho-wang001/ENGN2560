%> ------------------------------------------------------------------------
%> ENGN2560: Computer Vision
%>    Lab03: 3D Reconstruction and Bundle Adjustment
%> Problem2: Bundle Adjustment
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Load camera poses in world coordinate
load('data/Problem2/Rotation_View1.mat');       %> R1
load('data/Problem2/Rotation_View2.mat');       %> R2
load('data/Problem2/Rotation_View3.mat');       %> R3
load('data/Problem2/Translation_View1.mat');    %> T1
load('data/Problem2/Translation_View2.mat');    %> T2
load('data/Problem2/Translation_View3.mat');    %> T3

%> Load camera intrinsic matrix
load('data/Problem2/Intrinsic_Matrix.mat');     %> K

%> Load BA data
load('data/Problem2/BA_Data.mat');              %> BA_Data

%> Load 3D Points in world coordinate
load('data/Problem2/Points3D.mat');             %> Gamma_w

%> =====================================================
%> TODO: Do bundle adjustment (BA) on the given data
%> =====================================================

x0 = zeros(1, 765);

e1 = rotm2eul(R1);
e2 = rotm2eul(R2);
e3 = rotm2eul(R3);

x0(1:3) = e1(1:3);
x0(4:6) = e2(1:3);
x0(7:9) = e3(1:3);

x0(10:12) = T1(1:3)';
x0(13:15) = T2(1:3)';
x0(16:18) = T3(1:3)';

h = reshape(Gamma_w', 1, []);

x0(19:765) = h;

err = getReprojectionError(BA_Data, K, x0);

ff = @(x)(getReprojectionError(BA_Data, K, x));
options = optimoptions('lsqnonlin', 'Display', 'iter');
options.Algorithm = 'levenberg-marquardt';
BA_Output = lsqnonlin(ff, x0, [], [], options);

err1 = getReprojectionError(BA_Data, K, BA_Output);

function Error = getReprojectionError(BA_Data, K, x)
    e1 = x(1:3);
    e2 = x(4:6);
    e3 = x(7:9);

    T1 = x(10:12)';
    T2 = x(13:15)';
    T3 = x(16:18)';

    R1 = eul2rotm(e1);
    R2 = eul2rotm(e2);
    R3 = eul2rotm(e3);

    hh = x(19:765);

    GG = reshape(hh, 3, 249)';

    s = size(BA_Data, 1);
    
    e = 0;

    for i = 1:s
        if BA_Data(i, 1) == 1
            R = R1;
            T = T1;

            r1 = rp(R, T, K, GG(BA_Data(i, 2), :)');

            ee = ((r1(1) - BA_Data(i, 3))^2 + (r1(2) - BA_Data(i, 4))^2);

            e = e+ee;
        elseif BA_Data(i, 1) == 2
            R = R2;
            T = T2;

            r2 = rp(R, T, K, GG(BA_Data(i, 2), :)');

            ee = ((r2(1) - BA_Data(i, 3))^2 + (r2(2) - BA_Data(i, 4))^2);

            e = e+ee;
        else
            R = R3;
            T = T3;

            r3 = rp(R, T, K, GG(BA_Data(i, 2), :)');

            ee = ((r3(1) - BA_Data(i, 3))^2 + (r3(2) - BA_Data(i, 4))^2);

            e = e+ee;
        end
    end


    Error = e;
end

function r = rp(R, T, K, g)
    ga = R*g +T;
    c = K*ga;
    c = c./c(end);

    r = c;
end
