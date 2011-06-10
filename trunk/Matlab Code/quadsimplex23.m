function [alpha, b, iter, t] = quadsimplex22(P, T, C, ktype, lambda, giter, citer)
%[alpha, b, iter, t] = quadsimplex22(P, T, C, ktype, lambda, giter, citer)
%
% Work on larger working set size for gradient projection method.
%

	global m_P;
	global m_T;
	global m_C;
	global m_fcache;
	global upperfcache;
    global m_beta;
	global m_alpha;
	global m_ktype;
    global m_lambda;
    global m_iter;
    global Q;
    global m_svtype;
    global m_R;
    global idx_b;
    global idx_nb;
    global eps;
    
    global m_QCachedCol;
    
    Q = zeros(length(T), length(T));
    m_QCachedCol = false(1, length(T));
    

	% Allocate space for the support vectors
	m_alpha = zeros(length(T), 1);
	m_svtype = zeros(length(T), 1);
    
	
    % Initial gradient is equal to -Y
	m_fcache = -ones(length(T),1);
	upperfcache = zeros(length(T),1);
    

    %Initialize the global variables
	m_P = P;
	m_T = T;
	m_C = C;
	    
    m_ktype = ktype;
    m_lambda = lambda;
      
    m_iter = 0;
    
    m_beta = -m_T(1);
    updateCache;
    m_alpha(1) = 1e-12;
    m_svtype(1) = 1;
    m_R = sqrt(Q(1,1));
   
    idx_nb = 1;
    idx_b = 1;
    idx_b(1) = [];
    
    updateCache;
        
    tic; 
    eps = 1e-6;
    min_g = -1000;

    cycles = 1;

    

    if (citer > 0)
        m_C = 2^-15;
    end
    while (min_g < -eps || m_C < C)
        
        
        [min_g, idx] = min(m_fcache);

        disp(['iteration = ', num2str(m_iter), ' min_g = ', num2str(min_g), ' idx = ', num2str(idx), ...
              ' SVs = ', num2str(sum(m_svtype ~= 0))]);
          
        % Only do this step if there negative pricing variables. Otherwise,
        % one would be attempting to add a positive variable to the set of
        % non-bound SVs which are likely to destroy the guarantee of
        % non-singularity. 
        if (min_g < -eps)
            if (mod(cycles, giter) == 0)
                performGradProj(0);
                
%                 performGradProj(m_C*1000/(cycles.^2));
%                 performGradProj(.001);
            else
                % Perform pivoting on the pivot element
                takeStep(idx);
            end
        end
        
         if ( citer > 0 && min_g > -eps)
             
            pC = m_C;
            m_C = min(m_C * 2, C);
            
            m_alpha(m_svtype==2) = m_C;
            m_alpha(m_svtype==1) = m_alpha(m_svtype==1)/pC * m_C;
            
            upperfcache = upperfcache /pC * m_C;
            
            updateNBCache();
            FixComplementaryCondition();
            
            updateCache();
            
            [min_g, idx] = min(m_fcache);
            disp(['New C val = ',num2str(m_C)]);
            
        end
            
        cycles = cycles + 1;
        

%        disp(['iteration = ', num2str(m_iter), ' min_g = ', num2str(min_g), ...
%              ' SVs = ', num2str(sum(m_alpha > 0)), ' obj = ', num2str(-sum(m_alpha) - 0.5*m_alpha'*Q*m_alpha)]);

        


    end;
    
    t = toc

    alpha = m_alpha;
    iter = m_iter;
    
    % Note, look into why this variable needs to be negated.
    b = -m_beta;
    
    disp(['Num SVs = ', num2str(sum(m_svtype~=0))]);
    disp(['BSV = ', num2str(sum(m_svtype==2))]);
  
function val = getQD(is)

    persistent QD;
    global m_T;
    global m_P;
    global m_ktype;
    global m_lambda;

    if (isempty(QD)), 

        QD = zeros(size(m_T));

        for i = 1:length(m_T),
            if (m_ktype == 0),

                QD(i) = (-m_P(:,i)'*m_P(:,i)) * (m_T(i)*m_T(i));

            elseif (m_ktype == 1),

                QD(i) = -exp(-m_lambda * sum((m_P(:,i) - m_P(:,i)).^2,1))*m_T(i)*m_T(i);

            else

                error('Unrecognized kernel type');
            end;
        end

    end

    val = QD(is);
    
function val = getQ(i, j)
    
    global m_T;
    
    global m_P;
    global m_ktype;
    global m_lambda;  

    if (m_ktype == 0),

        val = (-m_P(:,i)'*m_P(:,j)) * (m_T(i)*m_T(j));

    elseif (m_ktype == 1),

        val = -exp(-m_lambda * sum((m_P(:,i) - m_P(:,j)).^2,1))*m_T(i)*m_T(j);

    else

        error('Unrecognized kernel type');
    end;
   
function QO = getQCache(i,j)

    global m_T;
    global Q;
    global m_P;
	global m_ktype;
    global m_lambda;
    global m_QCachedCol;
       
    for k = 1:length(j)
        if ~m_QCachedCol(j(k)) 

            if (m_ktype == 0),

                Q(:,j(k)) = (-m_P'*m_P(:,j(k))) .* (m_T'*m_T(j(k)));

            elseif (m_ktype == 1),

                Q(:,j(k)) = -exp(-m_lambda * sum((m_P(:,j(k))*ones(1,size(m_P,2)) - m_P).^2,1))*m_T(j(k)).*m_T;

            else
                error('Unrecognized kernel type');
            end;

            m_QCachedCol(j(k)) = true;

        end
    end
    
    QO = Q(i,j);
    

function performGradProj(mu)

    global m_T;
    global m_beta;
    global m_alpha;
    global m_C;
    global m_svtype;
    global m_fcache;
    global upperfcache;
    
    %
    % Note that this is really just a shortcut to computing the quantity
    % -Q(idx_nb,idxc)*m_alpha(idxc) which will result in far fewer
    % computations if the number of bound support vectors is significantly
    % higher than the number of non-bound support vectors.
    %
    %   delta_o = -1_o - beta * y_o + Q_os * alpha_s + Q_oc * alpha_c
    %
    % We are interested in, eventually, computing our change in the pricing
    % based upon the change in the set of non-support vectors versus bound
    % support vectors, i.e. 
    %
    %   Q_sc_new * alpha_c_new - Q_sc * alpha_c
    %
    % We can find Q_sc * alpha_c from our pricing cache without explicitly 
    % computing the pricing cache as follows:
    % 
    %  delta_s + 1_s + beta * y_s - Q_ss * alpha_s = Q_sc * alpha_c
    %
    % Note that we are computing using the anticipated pricing variable for 
    % the non-bound SVs here.. Note that we still have to consider the sign 
    % change on Q_os based upon how this was done in the original. 
    

    %
    % Here, we borrow a methodology similar to that presented by Joachims.
    % The steps are as follows:
    %
    g = m_fcache;
    g(m_svtype==2) = -g(m_svtype==2);
    g = g + m_beta * m_T';
    
    %
    % Note that d = -1 for alpha = 0 and KKT violator
    %           d =  1 for alpha = C and KKT violator
    %
    
    d = sign(g); 
    g = g .* m_T';
    
    [g,I] = sort(g,1,'ascend');
    
    Ns = 100000000;
    NMiss = 100000;
    %NMaxStep = 100000000;
   
    %
    % Take pairs from the top/bottom such that d < 1 for alpha = 0 and d >
    % 1 for alpha = C. We will search for no more than Ns pairs of points.
    % 
    
    idxtop = 1;
    idxbottom = length(g);
    check = zeros(size(g));
    idxfound = 0;
    
    adj = zeros(size(m_fcache));
    
    numMiss = 0;
    numInc = 0;
    idxfirst = 0;
    
    while (idxfound < Ns && numMiss < NMiss),
       
        while ((d(I(idxtop)) == 1 && m_svtype(I(idxtop)) == 0) || ...
               (d(I(idxtop)) == -1 && m_svtype(I(idxtop)) == 2) || ...
               (m_svtype(I(idxtop)) == 1) && ...
               g(idxtop) < 0),% && numInc < NMaxStep,                
            idxtop = idxtop + 1;     
        end;
        
        
        
        if (g(idxtop) >= 0 ), break; end;
        
        while ((d(I(idxbottom)) == 1 && m_svtype(I(idxbottom)) == 0) || ...
               (d(I(idxbottom)) == -1 && m_svtype(I(idxbottom)) == 2) || ...
               (m_svtype(I(idxtop)) == 1) && ...
                g(idxbottom) > 0),% && numInc < NMaxStep,                
            idxbottom = idxbottom - 1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        end;
        
        if (g(idxbottom) <= 0 ), break; end;
        
        % Determine if the pair will create a strictly decreasing objective
        idxs = [idxtop; idxbottom];
        
        numInc = numInc + 1;
               
        %
        % Test to ensure the pair of points will provide a strict decrease
        % in the objective function. The formula is as follows
        % 
        % d'*(Q_rs*alpha_s + Q_rc * alpha_c - 1_r - 1/2 * C*Q_rr*d) >= 0
        %
        % We observe that we can modify the formulas as follows:
        %
        % d'*(-d.*m_fcache + beta*y_r - 1/2 * C*Q_rr*d) >= 0 
        % 
        % where d_i = -1 if alpha_i = 0, d_i = 1 if alpha_i = C and
        % alpha_i' = alpha_i - d_i*C.
        
        
        Q12 = getQ(I(idxs(1)),I(idxs(2)));
        
        val = d(I(idxs))' * ( -d(I(idxs)).*(adj(I(idxs)) + m_fcache(I(idxs))) + m_beta * m_T(I(idxs))' + ...
                     0.5 * m_C * ...
                     [getQD(I(idxs(1))) Q12; ...
                      Q12               getQD(I(idxs(2))) ] ...
                      * d(I(idxs))) + mu;
                                 
%         disp(['val = ',num2str(val)]);
        if (val > 0)
            
%             disp(['Updating ', num2str(I(idxs(1))), ',', num2str(I(idxs(2)))]);
            if (idxfirst == 0)
                disp(['First entry found on ', num2str(idxtop)]);
                idxfirst = 1;
            end
            
            idxfound = idxfound + 1;
            check(I(idxs)) = 1;
            m_alpha(I(idxs)) = m_C - m_alpha(I(idxs));
            m_svtype(I(idxs)) = 2 - m_svtype(I(idxs));
                        
            % Compute an update to the error caching
            a1 = -getQCache(1:length(m_T),I(idxs)) * (-d(I(idxs)) * m_C);
            upperfcache = upperfcache + a1;
            
            a1(m_svtype==2) = -a1(m_svtype==2);
            a1(I(idxs)) = ((-2*(m_fcache(I(idxs))+adj(I(idxs)))) + a1(I(idxs)));
            adj = adj + a1;
                        
            numMiss = 0;
        
        else
            numMiss = numMiss + 1;
        end
        
        idxtop = idxtop + 1;
        idxbottom = idxbottom - 1;
        
    end

    
    disp(['Number of tests = ', num2str(numInc)]);
    
    % Adjust the error cache
    m_fcache = m_fcache + adj;
    

    if (idxfound > 0),          

        FixComplementaryCondition();
                   
        disp(['Num updates = ', num2str(idxfound)]);
        updateCache


    end
    
function FixComplementaryCondition()

    % Corrects situation where there is a non-zero value in m_fcache
    % (pricing variables) for the non-bounds SVs.
    
    global m_T;
    global m_beta;
    global m_alpha;
    global m_C;
    global m_R;
    global idx_nb;
    global m_svtype;
    global m_fcache;
    global eps;
    
    global upperfcache;
    
    % Now we perform steps identical to the takeStep() method with the same rhs.    
    % Update m_fcache for the non-bound SVs.

    h = zeros(size(m_T));
    [h(idx_nb), gb] = solveSub(getQCache(idx_nb,idx_nb), m_T(idx_nb), -m_fcache(idx_nb), 0);
    
      
    % Compute gamma
    gamma = -getQCache(idx_nb,idx_nb) * h(idx_nb)' - gb * m_T(idx_nb)';

    while (max(abs(m_fcache(idx_nb))) > eps)

%         m_iter = m_iter + 1;        
        idxh = find(h > 0);
        idxh2 = find(-h > 0);
        [theta, idxr] = min([m_alpha(idxh)'./h(idxh),(m_C - m_alpha(idxh2))'./-h(idxh2)]);

        t = [idxh, idxh2];
        idxr = t(idxr);


        if (~isempty(theta) && ( (theta < max(m_fcache(idx_nb)  ./ gamma)))),


           if (-h(idxr) > 0),
               m_svtype(idxr) = 2;
               upperfcache = upperfcache - getQCache(1:length(m_T),idxr)*m_C;
           else
               m_svtype(idxr) = 0;
           end;

           if (abs(h(idxr)) < eps)
               disp('Removal error');
           end


           % a value is leaving the basis
           m_beta = m_beta - theta * gb;
           m_alpha(idx_nb) = m_alpha(idx_nb) - theta * h(idx_nb)';
           m_fcache(idx_nb) = m_fcache(idx_nb) - theta * gamma;

           idxir = find(idx_nb == idxr);
           m_R = DownDateCholesky(m_R, m_T(idx_nb), idxir);
           idx_nb(idxir) = [];

        else

           theta = 1.0;
           m_beta = m_beta - theta * gb;
           m_alpha(idx_nb) = m_alpha(idx_nb) - theta * h(idx_nb)';
           m_fcache(idx_nb) = m_fcache(idx_nb) - theta * gamma;

           % We found a minimum, no value is leaving the basis

        end

        %
        % Now we need to drive the corresponding Lagrange to zero to force
        % the complementary conditions
        %
        if (max(abs(m_fcache(idx_nb))) <= eps), 
            break; 
        end;        

        h = zeros(size(m_T));                                
        [h(idx_nb),gb] = solveSub(getQCache(idx_nb,idx_nb), m_T(idx_nb), -m_fcache(idx_nb), 0);                        
        gamma = -getQCache(idx_nb,idx_nb) * h(idx_nb)' - gb * m_T(idx_nb)';
    end
        

function takeStep(idx)

	global m_T;
	global m_C;
	global m_fcache;
    global upperfcache;
	global m_beta;
	global m_alpha;
    global m_iter;
    
    global m_svtype;
    global m_R;
    global idx_nb;
    global eps;
    global h1;
    
    b = 0;
    bSlack = 0;
        
    h = zeros(size(m_T));
    gb = 0;
    gamma = 0;
      
    q = getQCache(idx_nb, idx);
        
    % We are adding a support vector, alpha is increased from zero.
    if (m_svtype(idx) == 0)
    
        %
        % We set the h value for the slack variable already in the data
        % set to 1.
        %
        h(idx) = -1;
        
        %
        % Solve sub-problem
        %
        [h(idx_nb), gb] = solveSub(getQCache(idx_nb,idx_nb), m_T(idx_nb), q, m_T(idx));
              
        % Compute gamma
        gamma = getQCache(idx, idx) - m_T(idx) * gb - getQCache(idx_nb, idx)' * h(idx_nb)';
        
    elseif (m_svtype(idx) == 2)
        
        
        bSlack = 1;
        
        %
        % We set the h value for the slack variable already in the data
        % set to 1.
        %
        h(idx) = 1;
        
        %
        % Solve sub-problem
        %
        [h(idx_nb), gb] = solveSub(getQCache(idx_nb,idx_nb), m_T(idx_nb), -q, -m_T(idx));
        gamma = getQCache(idx,idx) + m_T(idx) * gb + getQCache(idx_nb, idx)' * h(idx_nb)';

    end;
    
    m_svtype(idx) = 1;
    idx_nb = [idx_nb; idx];
    m_R = UpdateCholesky(m_R, -getQCache(idx_nb, idx_nb), m_T(idx_nb));
            
    if (bSlack)
        upperfcache = upperfcache + getQCache(1:length(m_T),idx)*m_C;
    end
    
    while (abs(m_fcache(idx)) > eps)
    
        m_iter = m_iter + 1;
                
        idxh = find(h > 0);
        idxh2 = find(-h > 0);
        [theta, idxr] = min([m_alpha(idxh)'./h(idxh),(m_C - m_alpha(idxh2))'./-h(idxh2)]);

        if (b == 0)
            h1 = theta;
            b = 1;
        end
        
        t = [idxh, idxh2];
        idxr = t(idxr);
               
		if (~isempty(theta) && ((gamma >= 0) || (theta < m_fcache(idx)/ gamma))),
        
                      
           if (-h(idxr) > 0),
               m_svtype(idxr) = 2;
               upperfcache = upperfcache - getQCache(1:length(m_T),idxr)*m_C;
           else
               m_svtype(idxr) = 0;
           end;
    
           idxir = find(idx_nb == idxr);
           m_R = DownDateCholesky(m_R, m_T(idx_nb), idxir);
           idx_nb(idxir) = [];
           
           % a value is leaving the basis
           m_beta = m_beta - theta * gb;
           m_alpha = m_alpha - theta * h';
           m_fcache(idx) = m_fcache(idx) - theta * gamma;
                      
           
		else
	           
           theta = m_fcache(idx) / gamma;
           m_beta = m_beta - theta * gb;
           m_alpha = m_alpha - theta * h';
           m_fcache(idx) = 0.0;
 
           % We found a minimum, no value is leaving the basis
           
		end
	

        %
        % Now we need to drive the corresponding Lagrange to zero to force
        % the complementary conditions
        %
        if (abs(m_fcache(idx)) <= eps), break; end;        
        
        h = zeros(size(m_T));
        
        e = zeros(length(m_T),1);
        e(idx) = 1;
        
                
        if (~bSlack)
                       
            [h(idx_nb),gb] = solveSub(getQCache(idx_nb,idx_nb), m_T(idx_nb), e(idx_nb), 0);                        
            gamma = -1;
            
        else
            
            [h(idx_nb),gb] = solveSub(getQCache(idx_nb,idx_nb), m_T(idx_nb), -e(idx_nb), 0);                                    
            gamma = -1;
            
        end;
        
        
    end;
   
    updateCache;
    
function m_R = UpdateCholesky(m_R, Q, T)
%  
%   Q, here, is the portion of the larger Q for the current non-bound support
%   vectors. The last row/column represents the row/column to be added. T
%   is the set of labels for the non-bound support vectors and the last
%   entry represents the entry to be added. 
% 
%  Update the Cholesky factorization by solving the following
%
%  R^T*r = -y_1 * y_n * Z^T * Q * e_1 + Z^T * q
%  r^T*r + rho^2 = e_1^T * Q * e_1 - 2 * y_1 * y_n * e_1^T * q + sigma
%

       
   Z = [-T(1) * T(2:end-1); eye(length(T) - 2) ] ;
   if (length(T) == 1), 
       m_R = sqrt(Q);
   elseif (length(T) == 2),
       m_R = sqrt([-T(1)*T(2) 1] * Q * [-T(1)*T(2); 1]);
   else
           
       q = Q(1:end-1,end);
       sigma = Q(end,end);

       r = m_R' \(-T(1)*T(end) * Z' * Q(1:end-1,1) + Z' * q) ;
       rho = sqrt(Q(1,1) - 2 * T(1) * T(end) * q(1) + sigma - r'*r);

        m_R = [m_R, r; 
               zeros(1,size(m_R,1)), rho];

   end
   
function m_R = DownDateCholesky(m_R, T, idx)

%
%   Here, we downdate the Cholesky factorization by removing the 
%   row/column, indexed by idx. There are two cases to consider. 
%   (1) if 2 <= idx <= end, remove the row, perform Givens rotations, and 
%   convert back to upper triangular and return the reduced R
%   (2) if idx = 1, apply the transformation A to R, then convert to upper
%   triangular and reduced the R to the new size. In this case, the
%   transform A is defined as 
%       [-y2 * [y3, ..., yn] 1; 
%               I            0];
%   

    if (idx > 1)

        m_R(:,idx-1) = [];

        for i = idx - 1: size(m_R, 1) - 1,
            m_R(i:i+1,:) = givens(m_R(i, i), m_R(i+1, i)) * m_R(i:i+1,:);
        end

        m_R = m_R(1:end-1,:);

    else

        A = [-T(2) * T(3:end), 1; 
             eye(length(T(3:end))), zeros(length(T(3:end)),1)];

        m_R = m_R * A;

        for i = 1:size(m_R,1) - 1,
            m_R(i:i+1,:) = givens(m_R(i, i), m_R(i+1, i)) * m_R(i:i+1,:);
        end

        m_R = m_R(1:end-1,1:end-1);        

    end
    

function [h, g] = solveSub(Q, y, q, r)
 
    global m_R;
            
    % Solve using the NULL space method.
    % Solve the following in sequence:
    %  1. y'Yhy = r for hy
    %      hy = y(1)r
    %      Y = [1; 0; 0; ...]
    %  2. Z'QZhz = Z'(q-QYhy) for hz
    %      R'R = -Z'QZ
    %      -R'Rhz = Z'(q-QYhy)
    %      rhs = Z'(q-QYhy)
    %      -R'Rhz = rhs
    %      -R'x = rhs
    %      Rhz = x
    %  3. h = Zhz + Yhy
    %  4. Y'Qh + Y'yg = Y'q for g
    %     Q(1,:)h + y(1)g = q(1)     
    %     
    if (length(y) > 1)
        Z = [-y(1)*y(2:end); eye(length(y)-1)];
        hy = r*y(1);
        
        rhs = Z'*(q-Q(:,1)*hy);
        
        
        hz = -m_R'\rhs;
        hz = m_R\hz;
        
        
        h = Z*hz;
        
        
        h(1) = h(1) + hy;
        
        g = y(1)*(q(1) - Q(1,:)*h);
    else
        % Just solve the following
        % 1. h = y(1)*r
        % 2. g = y(1)*(q - Q*h)
        %         
        h = y(1)*r;
        g = y(1)*(q - Q*h);
        
    end
    
function updateNBCache()
    
    % Doesn't assume that the non-bound cache is valid
    global m_alpha;
    global m_beta;
    global m_svtype;
    global m_fcache;
    global upperfcache;
    global m_T;
    
    idxs = find(m_svtype == 1);
    
    m_fcache(idxs) = -ones(length(idxs),1) - m_beta * m_T(idxs)'  - ...
                        getQCache(idxs, idxs) * m_alpha(idxs) + upperfcache(idxs); 
    
function updateCache()

global m_alpha;
global m_beta;
global m_svtype;
global m_fcache;
global upperfcache;

global m_T;

    
    m_fcache(:) = 0;
    
    idxzero = find(m_svtype == 0);
    idxs = find(m_svtype == 1);
    idxC = find(m_svtype == 2);
    
    m_fcache(idxzero) = -ones(length(idxzero),1) - m_beta * m_T(idxzero)'  - ...
                        getQCache(idxzero, idxs) * m_alpha(idxs) + upperfcache(idxzero); 
    
    m_fcache(idxC) = ones(length(idxC),1) + m_beta * m_T(idxC)' + ...
                        getQCache(idxC, idxs) * m_alpha(idxs) - upperfcache(idxC); 
                    
    
    