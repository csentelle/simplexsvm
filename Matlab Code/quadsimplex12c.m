function [alpha, b, iter, t, gcache, m] = quadsimplex12c(P, T, C, ktype, lambda)

	global m_P;
	global m_T;
	global m_C;
	global m_fcache;
	global m_beta;
	global m_alpha;
	global m_ktype;
    global m_lambda;
    global iter;
    global Q;
    global m_svtype;
    global m_R;
    global idx_b;
    global idx_nb;
    global eps;
    
    Q = zeros(length(T), length(T));
        
    N = length(T);
	if (ktype == 0),

        Q = P'*P .* (T'*T);
        
	elseif (ktype == 1),
        
        for i = 1:N,
            for j = 1:N,
	
                Q(i,j) = exp(-lambda * (P(:,i) - P(:,j))' * (P(:,i) - P(:,j))) * T(i) * T(j);
                
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
      
    iter = 0;
    
    m_beta = -m_T(1);
    updateCache;
    m_alpha(1) = 1e-12;
    m_svtype(1) = 1;
    m_R = sqrt(Q(1,1));
   
    idx_nb = 1;
    idx_b = 1;
    idx_b(1) = [];
    
    updateCache;
    
    % Select the pivot element (pricing)
    [min_g, idx] = min(m_fcache);
    
    tic; 
    eps = 1e-6;
	while (min_g < -eps)
        
        
        iter = iter + 1;
        
        % Perform pivoting on the pivot element
        takeStep(idx);

        [min_g, idx] = min(m_fcache);
        
        
        disp(['iteration = ', num2str(iter), ' min_g = ', num2str(min_g), ' SVs = ', num2str(sum(m_alpha > 0))]);
   
    end;
    
    t = toc

    alpha = m_alpha;

    % Note, look into why this variable needs to be negated.
    b = -m_beta;
    

function takeStep(idx)

	global m_T;
	global m_C;
	global m_fcache;
	global m_beta;
	global m_alpha;
    global iter;
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
        
%     disp(['Pivoting on idx = ', num2str(idx)]);
    
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
        gamma = -Q(idx, idx) - m_T(idx) * gb + Q(idx, idx_nb) * h(idx_nb)';

        
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
                                        
        gamma = -Q(idx,idx) + m_T(idx) * gb - Q(idx, idx_nb) * h(idx_nb)';

    end;
    
   % m_svtype(idx) = 1;
   % idx_nb = [idx_nb; idx];
    
    %disp(['Adding index to factorizaton ', num2str(idx)]);
    
    %m_R = UpdateCholesky(m_R, Q(idx_nb, idx_nb), m_T(idx_nb));
            
    bIndexAdded = false;
    
    while (abs(m_fcache(idx)) > eps)
    
        iter = iter + 1;
                
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

        if ~bIndexAdded, 
           m_svtype(idx) = 1;
           idx_nb = [idx_nb; idx];
           m_R = UpdateCholesky(m_R, Q(idx_nb, idx_nb), m_T(idx_nb));
           bIndexAdded = true;
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
                       
            [h(idx_nb),gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), -e(idx_nb), 0);                        
            gamma = -1;
                        
        else
            
            [h(idx_nb),gb] = solveSub(Q(idx_nb,idx_nb), m_T(idx_nb), e(idx_nb), 0);                                    
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
       
   if (length(T) == 1), 
       m_R(1,1) = sqrt(Q(1,1));
   elseif (length(T) == 2),       
       m_R(1,1) = sqrt(Q(1,1) + Q(2,2) - 2 * T(1) * T(2) * Q(1,2));
   else

        % Resize m_R
        m_R = [m_R, zeros(size(m_R,1),1); 
                   zeros(1,size(m_R,1)), 0];
       
       idx = length(T);
       prevSize = length(T) - 1;
              
       %
       % Solve the following system
       % Z = [-T(1) * T(2:end-1); eye(length(T) - 2) ] ;
       % r = m_R' \(-T(1)*T(end) * Z' * Q(1:end-1,1) + Z' * q) ;
       %
       
       % Form the RHS
       q = zeros(prevSize - 1, 1);
       for i = 1:prevSize - 1,
           q(i) = -T(1)*T(i+1)*(Q(1, 1) * -T(1)*T(idx) + Q(1,idx)) + ...
                   Q(i+1,1)*-T(1)*T(idx) + Q(i+1,idx);
       end
       
       
       %
       % Now solve the system through a forward substitution of 
       % triangular system
       % R' = x 0 0
       %      x x 0
       %      x x x
       %
       for i = 1:prevSize - 1,
           m_R(i,end) = q(i);
           for j = 1:i - 1,
               m_R(i,end) = m_R(i,end) - m_R(j,i) * m_R(j,end); 
           end;
           m_R(i,end) = m_R(i,end) / m_R(i,i);
       end

       
       for i = 1:prevSize-1,
           m_R(end,end)  = m_R(end,end) + m_R(i,end) * m_R(i,end);
       end

       
       m_R(end,end) = sqrt(Q(1,1) - 2 * T(1) * T(idx) * Q(1,idx) + Q(idx,idx) - m_R(end,end));

       
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
            m_R(i:i+1,:) = mygivens(m_R(i, i), m_R(i+1, i)) * m_R(i:i+1,:);
        end

        m_R = m_R(1:end-1,:);

    else

        %
        % Compute first row of m_R, in place. R(1,1) is the only element
        % that needs to be stored in a separate location.
        %
        R11 = m_R(1,1);
        for j = 1:size(m_R,2)-1, 
            m_R(1,j) = -T(2)*T(j+2) * R11 + m_R(1,j+1);
        end
        
        % Now shift all of the entries to the left.
        for i = 2:size(m_R,1),
            for j = 1:size(m_R,2) - 1,
                m_R(i,j) = m_R(i, j+1);
            end
        end
                                
        for i = 1:size(m_R,1) - 1,
            m_R(i:i+1,:) = mygivens(m_R(i, i), m_R(i+1, i)) * m_R(i:i+1,:);
        end

        m_R = m_R(1:end-1,1:end-1);        

    end

function G = mygivens(a, b)
    
    if (b == 0.0)
    
        c = 1.0; 
        s = 0.0;
       
    else
       
        if (abs(b) > abs(a))
               
            tau = -a / b;
            s = 1 / sqrt(1 + tau * tau); 
            c = s * tau;
           
        else
           
            tau = -b / a;
            c = 1 / sqrt(1 + tau * tau);
            s = c * tau;
        end
        
    end
    
    G = [c -s; s c];
        

function [h, g] = solveSub(Q, y, q, r)
 
    global m_R;
            
    % Solve using the NULL space method.
    % Solve the following in sequence:
    %  1. y'Yhy = r for hy
    %      hy = y(1)r
    %      Y = [1; 0; 0; ...]
    %  2. -Z'QZhz = Z'(QYhy+q) for hz
    %      R'R = -Z'QZ
    %      -R'Rhz = Z'(QYhy-q)
    %      rhs = Z'(QYhy-q)
    %      -R'Rhz = rhs
    %      -R'x = rhs
    %      Rhz = x
    %  3. h = Zhz + Yhy
    %  4. -Y'Qh + Y'yg = -Y'q for g
    %     -Q(1,:)h + y(1)g = -q(1)     
    %     
    if (length(y) > 1)
 
        % Solve for hy
        hy = r*y(1);
        
        % rhs = Z'*(q-Q(:,1)*hy);
        % Use implicit function to solve
        
        rhs = zeros(length(y)-1, 1);
        for i = 1:length(y) - 1,
            rhs(i) = -y(1) * y(i+1) * (-q(1) + Q(1,1)*hy) - q(i+1) + Q(i+1,1)*hy;
        end;
         
        % Perform backward, forward solve to obtain hz.
        %hz = -m_R'\rhs;
        %hz = m_R\hz;      
        hz = fwrdSolve(-m_R, rhs);
        hz = bkwrdSolve(m_R, hz);
        
        
        % Form the actual solution
        h = zeros(size(hz,1)+1,1);
        h(1) = -y(1)*y(2:end) * hz + hy;
        h(2:end) = hz;
        g = y(1)*(-q(1) + Q(1,:)*h);
        
    else
        % Just solve the following
        % 1. h = y(1)*r
        % 2. g = y(1)*(q - Q*h)
        %         
        h = y(1)*r;
        g = y(1)*(-q + Q*h);
    end
    
function x = fwrdSolve(R, b)

   %
   % Now solve the system through a forward substitution of 
   % triangular system
   % R' = x 0 0
   %      x x 0
   %      x x x
   %
   x = zeros(size(R,1),1);
   for i = 1:size(R,1),
       x(i) = b(i);
       for j = 1:i - 1,
           x(i) = x(i) - R(j,i) * x(j); 
       end;
       x(i) = x(i) / R(i,i);
   end

function y = bkwrdSolve(R, b)

   %
   % Now solve the system through a forward substitution of 
   % triangular system
   % R = x x x
   %     0 x x
   %     0 0 x
   %
   y = zeros(size(R,1),1);
   for i = size(R,1):-1:1,
       y(i) = b(i);
       for j = size(R,1):-1:i+1,
           y(i) = y(i) - R(i,j) * y(j); 
       end;
       y(i) = y(i) / R(i,i);
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
    
    m_fcache(idxzero) = -ones(length(idxzero),1) - m_beta * m_T(idxzero)'  + ...
                        Q(idxzero, idxs) * m_alpha(idxs) + Q(idxzero, idxC) * m_alpha(idxC);
    
    m_fcache(idxC) = ones(length(idxC),1) + m_beta * m_T(idxC)' - ...
                        Q(idxC, idxs) * m_alpha(idxs) - Q(idxC, idxC) * m_alpha(idxC);
    
    





