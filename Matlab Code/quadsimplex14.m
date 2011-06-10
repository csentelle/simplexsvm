function [alpha, b, iter, t, gcache, m] = quadsimplex14(P, T, C, ktype, lambda)

	global m_P;
	global m_T;
	global m_C;
	global m_fcache;
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
    
    Q = zeros(length(T), length(T));
        
    N = length(T);
	if (ktype == 0),

        Q = -P'*P .* (T'*T);
        
	elseif (ktype == 1),
        
        for i = 1:N,
            for j = 1:N,
	
                Q(i,j) = -exp(-lambda * (P(:,i) - P(:,j))' * (P(:,i) - P(:,j))) * T(i) * T(j);
                
            end;
        end;
        
	else
        error('Unrecognized kernel type');
	end;


	% Allocate space for the support vectors
	m_alpha = zeros(length(T), 1);
	m_svtype = zeros(length(T), 1);
    
	
    % Initial gradient is equal to -Y
	m_fcache = -ones(size(T));
	gcache = m_fcache;
    m = [];

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
    eps = 1e-3;
    min_g = -1000;

    cycles = 0;
    

    while (min_g < -eps )
        
        [min_g, idx] = min(m_fcache);
        
        if (mod(cycles, 20) == 1)
            
            performGradProj();
        else
            % Perform pivoting on the pivot element
            takeStep(idx);
        end
                      
        cycles = cycles + 1;
        
        disp(['iteration = ', num2str(m_iter), ' min_g = ', num2str(min_g), ...
              ' SVs = ', num2str(sum(m_alpha > 0)), ' obj = ', num2str(-sum(m_alpha) - m_alpha'*Q*m_alpha)]);

    end;
    
    t = toc

    alpha = m_alpha;
    iter = m_iter;
    % Note, look into why this variable needs to be negated.
    b = -m_beta;
    
function performGradProj()

    global m_T;
    global m_beta;
    global m_alpha;
    global m_C;
    global m_R;
    global Q;
    global idx_nb;
    global m_svtype;
    global m_fcache;
    global eps;
    global m_iter;
    
    Ns = 10;
    
    idx_nb = find(m_svtype == 1);
    
%     y = m_fcache'.*m_T';
%     y(m_svtype == 2) = -m_fcache(m_svtype == 2);
    grad = (-ones(length(m_T),1) - Q * m_alpha);
    Pr = eye(length(m_T)) - m_T'*m_T/(m_T*m_T');
    y = grad.*m_T';
    % Sort in ascending order
    [y,I] = sort(y,1,'descend');
        
    % Select items to modify
    d1 = ((-m_T(I)' == 1 & m_svtype(I) == 0) | (-m_T(I)' == -1 & m_svtype(I) == 2)) & y > 0;
    d2 = ((m_T(I)' == 1 & m_svtype(I) == 0) | (m_T(I)' == -1 & m_svtype(I) == 2)) & y < 0;
    
    idx1 = find(d1 == 1);
    idx2 = find(d2 == 1);
    
    idx1 = idx1(1:min(Ns,length(idx1)));
    idx2 = idx2(end-min(Ns, length(idx2))+1:end);
    
    % Now, we might have more of one side than the other, so we have to
    % take the minimum of each side.
    
    N = min(length(idx1),length(idx2));
    idx1 = idx1(1:N);
    idx2 = idx2(1:N);
 
    idx = I([idx1; idx2]);
    
    tmp = -Q(idx_nb, idx)*m_alpha(idx);
    disp(['obj before = ', num2str(-sum(m_alpha) - m_alpha'*Q*m_alpha)]);
    
    disp(['KKT Violators = ',num2str(sum(m_fcache < 0))]);
    disp(['Neg gradient ', num2str(sum(grad < 0))]);
    disp(['Neg gradient after project ', num2str(sum(Pr * grad < 0))]);
 
%    tmp = m_fcache(idx_nb)' + ones(size(idx_nb)) + m_T(idx_nb)'*m_beta + Q(idx_nb,idx_nb)*m_alpha(idx_nb);
    m_alpha(idx) = m_C - m_alpha(idx);
    m_svtype(idx) = 2 - m_svtype(idx);
   disp(['obj after = ', num2str(-sum(m_alpha) - m_alpha'*Q*m_alpha)]);

    if (abs(m_T * m_alpha) > 1e-6)
        error('Whoops');
    end;
    
    lenb = length(idx_nb);
    
    if (~isempty(idx))
        % Now we perform steps identical to the takeStep() method with the same rhs.    
        % Update m_fcache for the non-bound SVs.
        m_fcache(idx_nb) = -Q(idx_nb, idx) * m_alpha(idx) - tmp;

        h = zeros(size(m_T));
        [h(idx_nb), gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), -m_fcache(idx_nb)', 0);

        % Compute gamma
        gamma = -Q(idx_nb,idx_nb) * h(idx_nb)' - gb * m_T(idx_nb)';

        while (max(abs(m_fcache(idx_nb))) > eps)

            m_iter = m_iter + 1;        
            idxh = find(h > 0);
            idxh2 = find(-h > 0);
            [theta, idxr] = min([m_alpha(idxh)'./h(idxh),(m_C - m_alpha(idxh2))'./-h(idxh2)]);

            t = [idxh, idxh2];
            idxr = t(idxr);


            if (~isempty(theta) && ( (theta < max(m_fcache(idx_nb) ./ gamma')))),


               if (-h(idxr) > 0),
                   m_svtype(idxr) = 2;
               else
                   m_svtype(idxr) = 0;
               end;

               if (abs(h(idxr)) < 1e-3)
                   disp('Removal error');
               end


               % a value is leaving the basis
               m_beta = m_beta - theta * gb;
               m_alpha(idx_nb) = m_alpha(idx_nb) - theta * h(idx_nb)';
               m_fcache(idx_nb) = m_fcache(idx_nb) - theta * gamma';

               idxir = find(idx_nb == idxr);
               m_R = DownDateCholesky(m_R, m_T(idx_nb), idxir);
               idx_nb(idxir) = [];

            else

               theta = 1.0;
               m_beta = m_beta - theta * gb;
               m_alpha(idx_nb) = m_alpha(idx_nb) - theta * h(idx_nb)';
               m_fcache(idx_nb) = m_fcache(idx_nb) - theta * gamma';

               % We found a minimum, no value is leaving the basis

            end

            %
            % Now we need to drive the corresponding Lagrange to zero to force
            % the complementary conditions
            %
            if (max(abs(m_fcache(idx_nb))) <= eps), break; end;        

            h = zeros(size(m_T));                                
            [h(idx_nb),gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), -m_fcache(idx_nb)', 0);                        
            gamma = -Q(idx_nb,idx_nb) * h(idx_nb)' - gb * m_T(idx_nb)';

        end
    end
    
    disp(['Gradient Projection, found pairs = ', num2str(length(idx)), ' len(idx_nb) = ',num2str(lenb), ' -> ',num2str(length(idx_nb))]);
    disp(['obj after adjust = ', num2str(-sum(m_alpha) - m_alpha'*Q*m_alpha)]);

function takeStep(idx)

	global m_T;
	global m_C;
	global m_fcache;
	global m_beta;
	global m_alpha;
    global m_iter;
    global Q;
    global m_svtype;
    global m_R;
    global idx_nb;
    global eps;
    
    
    bSlack = 0;
        
    h = zeros(size(m_T));
    gb = 0;
    gamma = 0;
      
    q = Q(idx_nb, idx);
        
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
        [h(idx_nb), gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), q, m_T(idx));
                                                                  
        % Compute gamma
        gamma = Q(idx, idx) - m_T(idx) * gb - Q(idx, idx_nb) * h(idx_nb)';
        
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
        [h(idx_nb), gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), -q, -m_T(idx));
                                        
        gamma = Q(idx,idx) + m_T(idx) * gb + Q(idx, idx_nb) * h(idx_nb)';

    end;
    
    m_svtype(idx) = 1;
    idx_nb = [idx_nb; idx];
    m_R = UpdateCholesky(m_R, -Q(idx_nb, idx_nb), m_T(idx_nb));
            
    while (abs(m_fcache(idx)) > eps)
    
        m_iter = m_iter + 1;
                
        idxh = find(h > 0);
        idxh2 = find(-h > 0);
        [theta, idxr] = min([m_alpha(idxh)'./h(idxh),(m_C - m_alpha(idxh2))'./-h(idxh2)]);

        t = [idxh, idxh2];
        idxr = t(idxr);
               
		if (~isempty(theta) && ((gamma >= 0) || (theta < m_fcache(idx)/ gamma))),
        
                      
           if (-h(idxr) > 0),
               m_svtype(idxr) = 2;
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
        if (abs(m_fcache(idx)) <= eps) break; end;        
        
        h = zeros(size(m_T));
        
        e = zeros(length(m_T),1);
        e(idx) = 1;
        
                
        if (~bSlack)
                       
            [h(idx_nb),gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), e(idx_nb), 0);                        
            gamma = -1;
            
        else
            
            [h(idx_nb),gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), -e(idx_nb), 0);                                    
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
    
    
function updateCache()

global m_alpha;
global m_beta;
global m_svtype;
global m_fcache;
global m_T;
global Q;
    
    m_fcache(:) = 0;
    
    idxzero = find(m_svtype == 0);
    idxs = find(m_svtype == 1);
    idxC = find(m_svtype == 2);
    
    m_fcache(idxzero) = -ones(length(idxzero),1) - m_beta * m_T(idxzero)'  - ...
                        Q(idxzero, idxs) * m_alpha(idxs) - Q(idxzero, idxC) * m_alpha(idxC);
    
    m_fcache(idxC) = ones(length(idxC),1) + m_beta * m_T(idxC)' + ...
                        Q(idxC, idxs) * m_alpha(idxs) + Q(idxC, idxC) * m_alpha(idxC);
    
    