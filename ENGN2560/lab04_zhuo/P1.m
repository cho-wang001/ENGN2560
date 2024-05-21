function [a, b, I, C1, C2] = P1(PARAMS, Img1, Img2, K)
     

    if size(Img1, 3) == 3
        Img1 = rgb2gray(Img1);
    end
    
    if size(Img2, 3) == 3
        Img2 = rgb2gray(Img2);
    end
    
    [f1, d1] = vl_sift(single(Img1));
    [f2, d2] = vl_sift(single(Img2));
    
    [matches, scores] = vl_ubcmatch(d1, d2);
    [~, sortedIndices] = sort(scores);
    sortedMatches = matches(:, sortedIndices);

    mf1 = f1(1:2, sortedMatches(1,:));
    mf2 = f2(1:2, sortedMatches(2,:));
    
    [EE, II] = Ransac4Essential(PARAMS, mf1', mf2', K);

    %> Get the inliers which will be used to obtain the veridical (R,T)
    I = II;

    C1 = [];
    C2 = [];

    for i = I
        cc1 = [mf1(1, i), mf1(2, i), 1]';
        cc2 = [mf2(1, i), mf2(2, i), 1]';

        C1 = [C1, cc1];
        C2 = [C2, cc2];
    end
    
    [U, S, V] = svd(EE);
    W = [0, -1, 0; 1, 0, 0; 0, 0, 1];
    
    R1 = U * W * V';
    T1 = U(:, 3);
    R2 = U * W' * V';
    T2 = -U(:, 3);
    
    if det(R1) < 0 || det(R2) < 0
        EE = -1.*EE;
        [U, S, V] = svd(EE);
        W = [0, -1, 0; 1, 0, 0; 0, 0, 1];
        
        R1 = U * W * V';
        T1 = U(:, 3);
        R2 = U * W' * V';
        T2 = -U(:, 3);
    end
    
    s = size(C1, 2);
    invk = inv(K);
    
    gmf1 = [];
    gmf2 = [];
    
    for i = 1:s
        % rmf1 = invk*[C1(:,i); 1];
        rmf1 = invk*C1(:,i);
        gmf1 = cat(2, gmf1, rmf1);
    end
    
    for i = 1:s
        % rmf2 = invk*[C2(:,i); 1];
        rmf2 = invk*C2(:,i);
        gmf2 = cat(2, gmf2, rmf2);
    end
    
    s1 = 0;
    s2 = 0;
    s3 = 0;
    s4 = 0;
    
    for i = 1:s
        A = [-R1 * gmf1(:,i), gmf2(:,i)];
        b = T1;
        [Ua, Sa, Va] = svd(A, 'econ');
        rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
        rho1 = rhos(1);
        rho2 = rhos(2);
        
        if rho1>0 && rho2>0
            s1 = s1+1;
        end
    
    end
    
    for i = 1:s
        A = [-R1 * gmf1(:,i), gmf2(:,i)];
        b = T2;
        [Ua, Sa, Va] = svd(A, 'econ');
        rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
        rho1 = rhos(1);
        rho2 = rhos(2);
        
        if rho1>0 && rho2>0
            s2 = s2+1;
        end
    
    end
    
    for i = 1:s
        A = [-R2 * gmf1(:,i), gmf2(:,i)];
        b = T1;
        [Ua, Sa, Va] = svd(A, 'econ');
        rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
        rho1 = rhos(1);
        rho2 = rhos(2);
        
        if rho1>0 && rho2>0
            s3 = s3+1;
        end
    
    end
    
    for i = 1:s
        A = [-R2 * gmf1(:,i), gmf2(:,i)];
        b = T2;
        [Ua, Sa, Va] = svd(A, 'econ');
        rhos = Va * diag(1 ./ diag(Sa)) * Ua' * b;
        rho1 = rhos(1);
        rho2 = rhos(2);
        
        if rho1>0 && rho2>0
            s4 = s4+1;
        end
    
    end
    
    SS = [s1 s2 s3 s4];
    
    [~, index] = max(SS);
    
    if index==1
        disp('Veridical R and T are R1 and T1');
        RR = R1;
        TT = T1;
    elseif index==2
        disp('Veridical R and T are R1 and T2');
        RR = R1;
        TT = T2;
    elseif index==3
        disp('Veridical R and T are R2 and T1');
        RR = R2;
        TT = T1;
    else
        disp('Veridical R and T are R2 and T2');
        RR = R2;
        TT = T2;
    end
    
    a = RR;
    b = TT;
    
end

function [E, inlier_Idx] = Ransac4Essential(PARAMS,gamma1,gamma2,K)
    I = PARAMS.RANSAC_ITERATIONS;
    T = PARAMS.INLIER_THRESH;

    E = [];

    ml = 0;

    IK = inv(K);

    match_size = size(gamma1, 1);
    top_match_size = round(match_size*0.8);
    

    for i = 1:I
        c1m = [];
        c2m = [];
        
        sc = randperm(top_match_size, 5);
        
        for iii = 1:5
            g1 = IK*[gamma1(sc(iii),1); gamma1(sc(iii), 2); 1];
            g1 = g1';
            c1m = [c1m; g1];
        
            g2 = IK*[gamma2(sc(iii),1); gamma2(sc(iii), 2); 1];
            g2 = g2';
            c2m = [c2m; g2];
        end
        
        mi = cat(3, c1m, c2m);
        
        es = fivePointAlgorithmSelf(mi);
        
        if(size(es,3)~=0)
            for j = 1:size(es, 3)
                e = es{:,:,j};
                
                f = IK'*e*IK;
    
                l = 0;
                inlier_index = [];
    
                for n = 1:size(gamma1, 1)
                %for n = 1:top_match_size
            
    
                    c13d = [gamma1(n, 1); gamma1(n, 2); 1];
    
                    A = c13d(1, 1)*f(1,1) + c13d(2, 1)*f(1,2) + f(1,3);
                    B = c13d(1, 1)*f(2,1) + c13d(2, 1)*f(2,2) + f(2,3);
                    C = c13d(1, 1)*f(3,1) + c13d(2, 1)*f(3,2) + f(3,3);

                    distance = abs(A * gamma2(n, 1) + B * gamma2(n, 2) + C) / sqrt(A^2 + B^2);
    
                    if distance < T
                        l = l+1;
                        inlier_index = [inlier_index, n];
                    end
    
                end
    
                if l >= ml
                    ml = l;
                    E = e;
                    max_inlier_index = inlier_index;
                end
    
            end            
        end
        
    end
    

    inlier_Idx = max_inlier_index;

end
