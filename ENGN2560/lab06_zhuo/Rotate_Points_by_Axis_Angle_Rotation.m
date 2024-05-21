function Rotated_Points = Rotate_Points_by_Axis_Angle_Rotation(Rot_AxAng, Points3D)

%> Code Description: 
%     Given an axis-angle representation of a rotation matrix, rotate one or 
%     multiple 3D point(s).
%
%> Inputs: 
%     Rot_AxAng:      A 1x3 axis-angle representation of a regular rotation 
%                     matrix R. This can be achieved via
%                     Rot_AxAng = rotm2axang(R);
%                     Rot_AxAng = Rot_AxAng(end).*Rot_AxAng(1,1:3);
%     Points3D:       A 3xN matrix containing N 3D points.
%
%> Outputs:
%     Rotated_Points: A 3xN rotated 3D points.
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Mar. 10th, 2024

    Num_Of_Points3D = size(Points3D, 2);

    theta = dot(Rot_AxAng, Rot_AxAng);
    if theta > eps
        %> When theta is away from zero, use the Rodriguez formula
        %   P_  = P * cos(theta) + (w x P) * sin(theta) + w <w , P> (1 - cos(theta))
        %
        %> We want to be careful to only evaluate the square root if the
        %  norm of the angle_axis vector is greater than zero. Otherwise,
        %  we get a division by zero.

        theta = sqrt(theta);
        w = [Rot_AxAng(1)/theta, Rot_AxAng(2)/theta, Rot_AxAng(3)/theta]';
        w_cross_P = cross(repmat(w, 1, Num_Of_Points3D), Points3D, 1);
        w_dot_P = dot(repmat(w, 1, Num_Of_Points3D), Points3D, 1);
        Rotated_Points = zeros(3,Num_Of_Points3D);
        Rotated_Points(1,:) = Points3D(1,:).*cos(theta) + w_cross_P(1,:).*sin(theta) + w(1)*w_dot_P.*(1 - cos(theta));
        Rotated_Points(2,:) = Points3D(2,:).*cos(theta) + w_cross_P(2,:).*sin(theta) + w(2)*w_dot_P.*(1 - cos(theta));
        Rotated_Points(3,:) = Points3D(3,:).*cos(theta) + w_cross_P(3,:).*sin(theta) + w(3)*w_dot_P.*(1 - cos(theta));
    else
        %> If theta is near zero, the first order Taylor approximation of 
        %  the rotation matrix R corresponding to a vector w and angle w is
        %
        %  R = I + hat(w) * sin(theta)
        %
        %  But sin(theta) \sim theta and theta * w = angle_axis, which gives
        %
        %  R = I + hat(w)
        %
        %  and actually performing multiplication with P gives
        %  R * P = P + w x P.
        %
        %  Switching to the Taylor expansion near zero provides meaningful
        %  derivatives
        w_cross_P = cross(repmat(Rot_AxAng, 1, Num_Of_Points3D), Points3D, 1);
        Rotated_Points = Points3D + w_cross_P;
    end
end

