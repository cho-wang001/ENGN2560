function [abr, abt, np, nf, Point3D] = P3(PARAMS, Img2, Img3, K, R12, T12, cc1, cc2)


     


    ik = inv(K);
    s = size(cc1, 2);
    LWP = [];
    
    for i=1:s
        r1 = ik*cc1(:,i);
        r2 = ik*cc2(:,i);
    
        A = [ skew(r1) * [eye(3) [0 0 0]']; skew(r2) * [R12 T12]];
        [~, ~, V] = svd(A);
        lwp = V(:,end);
        lwp = lwp ./ lwp(end);

        lwp = lwp(1:3, 1);
    
        LWP = [LWP, lwp(1:3,1)];
    end
    
    

    
    
    
    if size(Img2, 3) == 3
        Img2 = rgb2gray(Img2);
    end
    
    if size(Img3, 3) == 3
        Img3 = rgb2gray(Img3);
    end
    
    [f2, d2] = vl_sift(single(Img2));
    [f3, d3] = vl_sift(single(Img3));
    
    [matches, scores] = vl_ubcmatch(d2, d3);
    [~, sortedIndices] = sort(scores);
    sortedMatches = matches(:, sortedIndices);
    mf2 = f2(1:2, sortedMatches(1,:));
    mf3 = f3(1:2, sortedMatches(2,:));
    
    
    
    [~, X, Y] = intersect(cc2(1:2,:)', mf2', 'rows');
    
    Point3D = LWP(:,X);
    Point2D = mf3(:,Y);
    
    
    [abr, abt] = Ransac4AbsPose(PARAMS, Point3D, Point2D, K);

    np = size(Point2D,2);
    nf = size(sortedMatches,2);

end





function [AbsR, AbsT] = Ransac4AbsPose(PARAMS, Points3D, Points2D,K)
    I = PARAMS.RANSAC_ITERATIONS;
    T = PARAMS.INLIER_THRESH;
    nm = size(Points3D, 2);
    max_inlier_count = 0;

    for i = 1:I
        
        sc = randperm(nm, 3);

        rn2 = inv(K)*[Points2D(:,sc); 1, 1, 1];

        rn3 = Points3D(:,sc);
        
        [rs, ts] = P3P_LambdaTwist(rn2, rn3);
        
        sss = size(rs, 3);
        inlier_count = 0;
        
        for j = 1:sss
            rt = rs(:,:,j);
            tt = ts(:,j);

            for f = 1:size(Points3D,2)
                rp2d = rp(rt, tt, K, Points3D(:,f));

                e = sqrt((rp2d(1) - Points2D(1,f))^2 + (rp2d(2) - Points2D(2,f))^2);
                if e < PARAMS.INLIER_THRESH
                    inlier_count = inlier_count + 1;
                end
            end

            if inlier_count > max_inlier_count
                max_inlier_count = inlier_count;
                ir = rt;
                it = tt;
            end

        end
        
    end

    AbsR = ir;
    AbsT = it;
end




function S = skew(v)

    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end

function r = rp(R, T, K, g)
    ga = R*g +T;
    c = K*ga;
    c = c./c(end);

    r = c;
end