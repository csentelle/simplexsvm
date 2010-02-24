#include ".\svm.h"
#include <algorithm>
#include "time.h"
#include <functional>
#include "stdlib.h"

#undef PROFILE
//#define PROFILE	&Profile

#include "hwprof.h"

CHWProfile Profile; 


template <class T>
struct IndexCompare
{
    IndexCompare(const T& arr) : m_arr(arr) {};
    ~IndexCompare() {};
    
    bool operator()(const size_t idx1, const size_t idx2)
    {
        return m_arr(idx1) < m_arr(idx2);
    }

    const T& m_arr;
};


SVM::SVM(Kernel& kernel, double C, double tol, ProxyStream& os)
: m_kernel(kernel)
, m_C(C)
, m_tol(tol)
, m_os(os)
, m_RStorage(1,1,ColumnMajorArray<2>())
, m_R(1,1,ColumnMajorArray<2>())
{

}

SVM::~SVM(void)
{

}

void SVM::train(const Array<double, 2>& P, 
                const Array<double, 1>& T,
                Array<double, 1>& alpha,
                double& b,
                int& iter,
                double& t,
                int working_size,
                int shrinking_iter)
{

	Array<double, 1> fcache(T.size());
    Array<double, 1> upperfcache(T.size());

	double beta = 0.0;

    //
    // Allocate temporary arrays, these will be used in some functions
    // to prevent heap allocation multiple times, for efficiency.
    //

    m_hS.resize(fcache.shape());
    m_etS.resize(fcache.shape());
    m_qS.resize(fcache.shape());

    m_RStorage.resize(STORAGE_NEW_INCREMENT, STORAGE_NEW_INCREMENT);
    m_R.reference(m_RStorage(Range(0,0), Range(0,0)));


    m_status.resize(fcache.shape());
    m_idxpossible.reserve(fcache.size());
    m_idxreplace.reserve(fcache.size());
   
	//
	// Initialize the arrays
	//
	fcache(Range::all()) = -1;
    upperfcache(Range::all()) = 0.0;
	
	alpha(Range::all()) = 0.0;

    m_status(Range::all()) = 0;

    for (int i = 0; i < T.size(); i++)
        m_fcache_indices.push_back(i);

    if ((working_size > fcache.size()) || (working_size <= 0))
        working_size = fcache.size();

    m_os << infolevel(7) << "P = " << endl << P << endl;
    m_os << infolevel(7) << "T = " << endl << T << endl;

    int initS = 0; 
    beta = -T(initS);

	alpha(initS) = 1e-12;
    m_idxnb.push_back(initS);
    m_status(initS) = 1;

    reinitializeCache(fcache, upperfcache, T, alpha, beta);

    m_R(0, 0) = sqrt(m_kernel.getQss(initS));


    m_os << infolevel(7) << "fcache = " << fcache << endl;

	//
    // Find the next index to enter the cache. Note that the 
    // minIndex function returns a TinyVector, of which we 
    // want the first value.
    //
    int idx = minIndex(fcache)(0);
    double min_g = min(fcache);

	Profile.reset();

    iter = 0;
    time_t ts = clock();
	
	while (min_g < -m_tol)
    {
               
        m_os << infolevel(1) << "iter = " << iter 
                             << " min_g = " << min_g 
                             << " idx = " << idx << endl;


        // Perform pivoting on the pivot element
        takeStep(alpha, idx, fcache, T, beta, upperfcache, iter);


		idx = updateCacheStrategy(working_size, T, alpha, beta, fcache, upperfcache, min_g);
        
		// 	If we appear to have hit the stopping criterion, then, proceed with
		//  first withi a reinitialization of a working set corresponding to the 
		//  partial pricing technique. If this doesn't work, then we need to 
		//  recompute the entire error cache for sprinting purposes. m_workset
		//  is the set of indices currently being worked on by the partial 
		//  pricing technique and m_fcache_indices are the current set of indices
		//  being worked on for shrinking.
        if (min_g > -m_tol)
        {
            
            m_workset.erase(m_workset.begin(), m_workset.end());
            
            idx = updateCacheStrategy(working_size, 
                                      T, 
                                      alpha, 
                                      beta, fcache, upperfcache, min_g);

            if (min_g > -m_tol)
            {
                m_fcache_indices.erase(m_fcache_indices.begin(),
                                       m_fcache_indices.end());

                for (int i = 0; i < fcache.size(); i++)
                    m_fcache_indices.push_back(i);

                m_workset.erase(m_workset.begin(), m_workset.end());
                
                idx = updateCacheStrategy(working_size, 
                                        T, 
                                        alpha, 
                                        beta, fcache, upperfcache, min_g);

            }
        }

        // Perform shrinking
        else if (iter % shrinking_iter == 0)
        {

            reinitializeCache(fcache, upperfcache, T, alpha, beta);

            for (int i = 0; i < m_fcache_indices.size(); i++)
            {

                if (fcache(m_fcache_indices[i]) > -m_tol)
                {
                    m_fcache_indices.erase(remove(m_fcache_indices.begin(),
                                                  m_fcache_indices.end(),
                                                  m_fcache_indices[i]),
                                           m_fcache_indices.end());

                }
            }
        }
    }


	//Profile.dumpprint(m_os);
    t = ((double)clock() - (double)ts) / (double)CLOCKS_PER_SEC;

    // Compute B
    // We are computing B according to:    
    b = beta;

	//////END_PROF();
	Profile.dumpprint(m_os, 0);
	Profile.reset();
}

void SVM::takeStep(Array<double, 1>& alpha, 
              const int idx,
              Array<double, 1>& fcache,
              const Array<double, 1>& T,
              double& beta, 
              Array<double, 1>& upperfcache,
			  int& iter)
{
    double eps = 1e-15;
    bool bSlack = false;
    double gamma = 0.0;
    int N = T.size();

    Array<double, 1> h = m_hS(Range(0, (int)m_idxnb.size())); 

	double gb = 0.0;

    for (int kidx = 0; kidx < h.size(); kidx++)
        h(kidx) = 0.0;

    // Find the indices for bound, non-bound support vectors. Note that 
    // this, in reality, needs to be moved to another section.
	Array<double, 1> q = m_qS(Range(0, (int)m_idxnb.size() - 1));

    m_os << infolevel(2) << "Pivoting on index: " << idx << endl;

    for (int i = 0; i < q.size(); i++)
    {
        q(i) = m_kernel(idx, m_idxnb[i]);
    }

    m_os << infolevel(4) << "q = " << q << endl;

    // First, we are adding a new variable to the basis, identified by idx    
    // We are adding a support vector, alpha is increased from zero. 

    m_os << infolevel(4) << "m_status = " << m_status(idx) << endl;

    if (m_status(idx) == 0)
    {    

        m_os << infolevel(3) << "Increasing alpha from 0" << endl;

        //
        // Solve sub-problem
        //
        solveSubProblem(m_R, 
                        q,
                        T(idx),
                        T,
                        h,
                        gb);

        
        //       
        //  We set the h value for the slack variable already in the data
        //  set to 1.
        //        
        h(h.size() - 1) = -1;


        m_os << infolevel(4) << "Calculations for g, h" << endl;
        m_os << infolevel(4) << "gb = " << gb << endl;
        m_os << infolevel(4) << "ht = " << h << endl;


        //
        // Compute gamma. Note that we don't actually need the term
        // for g(idx + 1) since this will be zero as idx represents
        // a non-support vector and g is only non-zero for bound 
        // support vectors (m_status == 2)
        //
        gamma = -m_kernel.getQss(idx) - T(idx) * gb; 

        for (int k = 0; k < m_idxnb.size(); k++)
        {
			gamma += m_kernel(m_idxnb[k], idx) * h(k);
        }
        
        m_os << infolevel(4) << "gamma = " << gamma << endl;

    }            
    else if (m_status(idx) == 2)
    {
     
        m_os << infolevel(3) << "alpha(idx) = C" << endl;


        bSlack = true;
                
                
        // Solve sub-problem
		q = -q;
        solveSubProblem(m_R, 
                        q,
                        -T(idx),
                        T,
                        h,
                        gb);
        q = -q;

        // We set the h value for the slack variable already in the data
        // set to 1.
        h(h.size() - 1) = 1;


        m_os << infolevel(4) << "Calculations for g, h" << endl;
        m_os << infolevel(4) << gb << endl;
        m_os << infolevel(4) << h << endl;
        
   
        gamma = -m_kernel.getQss(idx) + T(idx) * gb; 
               
        for (int k = 0; k < (int)m_idxnb.size(); k++)
        {
            gamma -= m_kernel(m_idxnb[k],idx) * h(k);
        }

		m_os << infolevel(4) << "gamma = " << gamma << endl;
       
    }
    
    bool bAddedIndex = false;

	m_os << infolevel(5) << "New Chol Factorization = " << endl << m_R << endl;
                
    while (abs(fcache(idx)) > m_tol)
    {
        // 
        // Search for the minimum theta. 
        // Augment the h vector for non-bound support vectors        
        //
        iter++;
        double theta = 1e300;
        int idxr = -1;
        int idxh = -1;
        int idxt = -1;

        for (int i = 0; i < h.size(); i++)
        {
            //
            // The first size() - 1 entries of the h-vector correspond to the 
            // non-bound support vectors and the last entry corresponds to the
            // h value for the variable entering the basis (passed to this function).
            //
            if (i < h.size() - 1)
                idxt = m_idxnb[i];
            else
                idxt = idx;


            m_os << infolevel(6) << "idxt = " << idxt << endl;
            m_os << infolevel(6) << "h(i) = " << h(i) << endl;

            if (h(i) > 0)
            {
                double thetaTmp = alpha(idxt) / h(i);
                if ( thetaTmp + eps < theta) 
                {
                    m_os << infolevel(6) << "theta = thetaTmp = " << setprecision(12) << thetaTmp << endl;
                    m_os << infolevel(6) << "idxh = " << i << endl;
                    m_os << infolevel(6) << "idxr = " << idxt << endl;

                    theta = thetaTmp;

                    idxh = i;
                    idxr = idxt;
                }
            }
            else if (-h(i) > 0)
            {
                double thetaTmp = (m_C - alpha(idxt) ) / -h(i);
                if (thetaTmp + eps < theta)
                {
                    m_os << infolevel(6) << "theta = thetaTmp = " << thetaTmp << endl;
                    m_os << infolevel(6) << "idxh = " << i << endl;
                    m_os << infolevel(6) << "idxr = " << idxt << endl;

                    theta = thetaTmp;
                    idxh = i;
                    idxr = idxt;
                }
            }
        }

        m_os << infolevel(2) << "theta = " << theta << endl;
        m_os << infolevel(2) << "fcache(idx) / gamma = " << fcache(idx) / gamma << endl;

		// Note that we cannot use the "tol", here, for this decision, but a much tighter
		// eps, otherwise cycling can occur. 
		if ( (gamma > -eps) || (theta < fcache(idx)/ gamma - eps))
        {

            m_os << infolevel(2) << "Value leaving basis: " << idxr << endl;

            // a value is leaving the basis
            beta = beta - theta * gb;

			if (!bAddedIndex)
            {
                for (int k = 0; k < h.size() - 1; k++)
                    alpha(m_idxnb[k]) = alpha(m_idxnb[k]) - theta * h(k);

                alpha(idx) = alpha(idx) - theta * h(h.size() - 1);
            }
            else
            {
                for (int k = 0; k < h.size(); k++)
                    alpha(m_idxnb[k]) = alpha(m_idxnb[k]) - theta * h(k);
            }


            fcache(idx) = fcache(idx) - theta * gamma;

               
            m_os << infolevel(7) << "beta = " << beta << endl;
            m_os << infolevel(7) << "alpha = " << alpha << endl;

            // If we added to the bound set, that is we hit alpha = C.
            if (-h(idxh) > 0)
            {

                m_idxb.push_back(idxr);
                m_status(idxr) = 2;

                // Update the upper bound cache
                for (int k = 0; k < upperfcache.size(); k++)
                    upperfcache(k) += m_kernel(idxr,k) * m_C;

            }
            else
            {
                m_status(idxr) = 0;
            }

            if ( (idxr != idx) || 
                ((idxr == idx) && (bAddedIndex == true)) )
            {

				//
				// Now we need to find the item to remove from the basis
				// as well as figure out a new factorization.
				//
				vector<int>::iterator vIter = 
					find(m_idxnb.begin(), m_idxnb.end(), idxr);
	            
				int idx_pos = (int)(vIter - m_idxnb.begin());

				m_os << "idx_pos = " << idx_pos << endl;

				// 
				// Update the factorization by removing the corresponding, row/column, from 
				// the factorization. 
				//
				reduceCholFactor(T, idx_pos);

				m_os << infolevel(5) << "Reduced Cholesky = " << endl;
				m_os << infolevel(5) << m_R << endl;            

				// Erase the value from the non-bound values
				if (vIter != m_idxnb.end())
				{
					if (m_status(*vIter) == 1)
						m_status(*vIter) = 0;

					m_idxnb.erase(vIter);
	               
				}
				else
				{
					m_os << warning() << "Invalid index removed" << endl;
				}
			}
           
        }
		else
        {
	
            idxr = -1;
            m_os << infolevel(2) << "Found minimum of objective." << endl;
           
            theta = fcache(idx) / gamma;
            
            // a value is leaving the basis
            beta = beta - theta * gb;

			if (!bAddedIndex)
            {
                for (int k = 0; k < h.size() - 1; k++)
                    alpha(m_idxnb[k]) = alpha(m_idxnb[k]) - theta * h(k);

                alpha(idx) = alpha(idx) - theta * h(h.size() - 1);
            }
            else
            {
                for (int k = 0; k < h.size(); k++)
                    alpha(m_idxnb[k]) = alpha(m_idxnb[k]) - theta * h(k);
            }
            
            fcache(idx) = 0.0;

            m_os << infolevel(7) << "beta = " << beta << endl;
            m_os << infolevel(7) << "alpha = " << alpha << endl;
           
        }           

        if (bAddedIndex == false)
        {

            if (bSlack)
            {
                // In anticipation, go ahead and remove this from the bound 
                // set.
                m_idxb.erase(remove(m_idxb.begin(), 
                                    m_idxb.end(), 
                                    idx), 
                             m_idxb.end());

                // Remove values from upper cache
                for (int k = 0; k < upperfcache.size(); k++)
                    upperfcache(k) -= m_kernel(idx,k) * m_C;


            }

            m_os << " idx = " << idx << " idxr = " << idxr << endl;

            if (idx != idxr)
            {

                // Update the cholesky factorization
                addToCholFactor(T, idx);

				//
                // Go ahead and add to the non-bound support vector list. This has to be done after adding 
                // to the factorization, otherwise, the factorization will be incorrect.
                //
                m_idxnb.push_back(idx);
                m_status(idx) = 1;
                    
                m_os << infolevel(5) << "New Chol Factorization = " << endl << m_R << endl;
            }
        
            bAddedIndex = true;
        }


		//
        // Now we need to drive the corresponding Lagrange to zero to force
        // the complementary conditions
        //
        if (abs(fcache(idx)) <= 1e-6) 
            break; 
        
        h.reference(m_hS(Range(0, (int)m_idxnb.size() - 1)));
    
        if (!bSlack)
        {
            m_os << infolevel(2) << "~bSlack" << endl;           
            
            // Solve sub-problem
            
            Array<double, 1> et = m_etS(Range(0, h.size() - 1));

            for (int i = 0; i < m_idxnb.size(); i++)
			{
				et(i) = m_idxnb[i] == idx ? -1 : 0;
			}

            m_os << infolevel(4) << "et = " << et << endl;
           
            solveSubProblem(m_R, 
                            et,
                            0,
                            T,
                            h,
                            gb);
            
            m_os << infolevel(4) << "h before = " << h << endl;
            // Add the solution back into the larger matrix

            m_os << infolevel(4) << "gb = " << gb << endl;
                        
            // Compute gamma
            gamma = -1;
            m_os << infolevel(4) << "h = " << h << endl;
        }
        else
        {   

            m_os << infolevel(2) << "bSlack" << endl;

            // Solve sub-problem            
            Array<double, 1> et = m_etS(Range(0, h.size() - 1));

            for (int i = 0; i < m_idxnb.size(); i++)
			{
				et(i) = m_idxnb[i] == idx ? 1 : 0;
			}


            solveSubProblem(m_R, 
                            et,
                            0,
                            T,
                            h,
                            gb);
            
            // Add the solution back into the larger matrix            
            m_os << infolevel(4) << gb << endl;

            gamma = -1;

            m_os << infolevel(4) << "gb = " << gb << endl;
            m_os << infolevel(4) << "h = " << h << endl;
            
        } // 
        
    } // while

}


int SVM::updateCacheStrategy(const int nWorkingSize, 
                             const Array<double, 1>& T,
                             const Array<double, 1>& alpha, 
                             const double beta,
                             Array<double, 1>& fcache,
                             Array<double, 1>& upperfcache,
                             double& ming)
{

    int idxmin = 0;

    if (m_workset.empty())
    {


        reinitializeCache(fcache, upperfcache, T, alpha, beta);

		//
		// It is possible that the process of shrinking has removed a number
		// of items from the cache list where the number of remaining points
		// is smaller than the working set size.
		//
        
		if (m_fcache_indices.size() > nWorkingSize)
		{
			//
			// Sort, in ascending order, the first set of items that are necessary
			//
			partial_sort(m_fcache_indices.begin(), 
						 m_fcache_indices.begin() + nWorkingSize,
						 m_fcache_indices.end(), 
						 IndexCompare<Array<double,1> >(fcache));

			// Assign the first elements as the working set
			m_workset.assign(m_fcache_indices.begin(), 
							 m_fcache_indices.begin() + nWorkingSize);

			// Resort the indices back to the way they were. We no longer need
			// the sorted indices.
			sort(m_fcache_indices.begin(), m_fcache_indices.end());

		}
		else
		{
			// Just assign the current remaining indices as the working set
			m_workset.assign(m_fcache_indices.begin(), 
				             m_fcache_indices.end());
		}

        // Sort the indices
        sort(m_workset.begin(), m_workset.end());

		

    }
    else
    {


		// Create a temporary vector to hold some of the 
		// kernel values for performance reasons.
		vector<double*> vctr(m_idxnb.size());
		for (int i = 0; i < m_idxnb.size(); i++)
			vctr[i] = m_kernel[m_idxnb[i]];
        

        for (int i = 0; i < m_workset.size(); i++)
        {
            int idx = m_workset[i];

            if (m_status(idx) == 0)
            {
            
                double cache = -1 - T(idx) * beta;
                for (int k = 0; k < m_idxnb.size(); k++)
					cache += vctr[k][idx] * alpha(m_idxnb[k]);
                cache += upperfcache(idx);

                fcache(idx) = cache;

            }
            else if (m_status(idx) == 2)
            {

                double cache =  -1 - T(idx) * beta;
                for (int k = 0; k < m_idxnb.size(); k++)           
                    cache += vctr[k][idx] * alpha(m_idxnb[k]);

                cache += upperfcache(idx);

                fcache(idx) = -cache;
            }
            else
            {
                fcache(idx) = 0.0;
            }

        }
    }

    // Find the minimum in the workset.
    idxmin = 0;
    ming = 1e300;

	for (int i = 0; i < m_workset.size(); i++)
    {
        if (fcache(m_workset[i]) < ming)
        {
            ming = fcache(m_workset[i]);
            idxmin = m_workset[i];
        }
    }

    return idxmin;

}



void SVM::reinitializeCache(Array<double, 1>& fcache,
                            Array<double, 1>& upperfcache,
                            const Array<double, 1>& T,
							const Array<double, 1>& alpha,
							const double beta)
{

	vector<double*> vctr(m_idxnb.size());
	for (int i = 0; i < m_idxnb.size(); i++)
		vctr[i] = m_kernel[m_idxnb[i]];

    for (int i = 0; i < m_fcache_indices.size(); i++)
    {
        int idx = m_fcache_indices[i];

        if (m_status(idx) == 1)
        {
            fcache(idx) = 0.0;   
        }
        else if (m_status(idx) == 2)
        {

                // See if we can calculate pi, on our own.
                double cache =  1 + T(idx) * beta;
                for (int k = 0; k < m_idxnb.size(); k++)
				    cache -= vctr[k][idx] * alpha(m_idxnb[k]);

                cache -= upperfcache(idx);

                fcache(idx) = cache;
            
        }
        else if (m_status(idx) == 0)
        {
            double cache = -1 - T(idx) * beta;
            for (int k = 0; k < m_idxnb.size(); k++)
				cache += vctr[k][idx] * alpha(m_idxnb[k]);

            cache += upperfcache(idx);
            
            fcache(idx) = cache;
        }
    }

}

void SVM::solveSubProblem(const Array<double, 2>& R, 
                          const Array<double, 1>& qs,
                          const double ys,
                          const Array<double, 1>& T,
                          Array<double, 1>& h,
                          double& g)
{

    //BEGIN_PROF("solveSubProblem");
    //
    // Here, we are going to use a Cholesky factorization and the Shur complement to 
    // solve the system of equations associated with the sub-problem
    //

    // Solve using the NULL space method.
    // Solve the following in sequence:
    //  1. y'Yhy = r for hy
    //      hy = y(1)r
    //      Y = [1; 0; 0; ...]
    //  2. -Z'QZhz = Z'(QYhy+q) for hz
    //      R'R = -Z'QZ
    //      -R'Rhz = Z'(QYhy-q)
    //      rhs = Z'(QYhy-q)
    //      -R'Rhz = rhs
    //      -R'x = rhs
    //      Rhz = x
    //  3. h = Zhz + Yhy
    //  4. -Y'Qh + Y'yg = -Y'q for g
    //     -Q(1,:)h + y(1)g = -q(1)     
    //     

	int N = qs.size();

	if (N > 1)
	{
		// Solve for hy = ys * y(1) 
		double hy = ys * T(m_idxnb[0]);
        
        // rhs = Z'*(q - Q(:,1)*hy);
        // Use implicit function to solve
		Array<double,1> hz(N - 1);

		// Compute the rhs for solving R'Rhz = rhs. Note that we compute the negative of the rhs at this point 
		// in anticipation of computing -R'hz = rhs
		for (int i = 0; i < N - 1; i++)
		{
			assert(i >= 0 && i < N - 1);
            hz(i) = T(m_idxnb[0]) * T(m_idxnb[i + 1]) * (m_kernel.getQss(m_idxnb[0]) * hy - qs(0)) +
				     qs(i + 1) - m_kernel[m_idxnb[i + 1]][m_idxnb[0]] * hy;
		}

        // Perform backward, forward solve to obtain hz.
        // hz = -m_R'\rhs;
        // hz = m_R\hz;      
        hz = fwrdSolve(m_R, hz);
        hz = bkwrdSolve(m_R, hz);
        
        // Form the actual solution
		h(0) = hy;
		for (int i = 1; i < N; i++)
		{
		    assert(i - 1 >= 0 && i - 1 < N - 1);
			h(0) -= T(m_idxnb[0]) * T(m_idxnb[i]) * hz(i - 1);
			h(i) = hz(i - 1);
		}

		g = -T(m_idxnb[0]) * qs(0);
		for (int i = 0; i < N; i++)
		{
			g += T(m_idxnb[0]) * m_kernel[m_idxnb[0]][m_idxnb[i]] * h(i);
		}
	}
    else
	{
        // Just solve the following
        // 1. h = y(1)*r
        // 2. g = y(1)*(q - Q*h)
        //         
        h(0) = T(m_idxnb[0]) * ys;
        g = T(m_idxnb[0]) * (m_kernel.getQss(m_idxnb[0]) * h(0) - qs(0));
	}
}

inline Array<double, 1>& SVM::fwrdSolve(const Array<double, 2>& R, Array<double, 1>& x)
{
   //
   // Now solve the system through a forward substitution of 
   // triangular system
   // R' = x 0 0
   //      x x 0
   //      x x x
   //
  
   for (int i = 0; i < R.extent(0); i++)
   {
	   for (int j = 0; j <= i - 1; j++)
	   {
           assert(i >= 0 && j >= 0 && i < x.extent(0) && j < x.extent(0));
		   x(i) = x(i) - R(j, i) * x(j); 
	   }
       x(i) = x(i) / R(i,i);
   }

   return x;
}

inline Array<double, 1>& SVM::bkwrdSolve(const Array<double, 2>& R, Array<double, 1>& x)
{
   //
   // Now solve the system through a forward substitution of 
   // triangular system
   // R = x x x
   //     0 x x
   //     0 0 x
   //
   for (int i = R.extent(0) - 1; i >= 0; --i)
   {
	   for (int j = R.extent(0) - 1; j >= i + 1; --j)
	   {
           assert(i >= 0 && j >= 0 && i < x.extent(0) && j < x.extent(0));
           x(i) = x(i) - R(i,j) * x(j); 
	   }
	   x(i) = x(i) / R(i,i);
   }
   
   return x;
}

inline void computeGivens(const double a, 
                     const double b, 
                     double& c, 
                     double& s)
{
    double tau = 0.0;

    if (b == 0.0)
    {
        c = 1.0; 
        s = 0.0;
    }   
    else
    {   
        if (abs(b) > abs(a))
        {       
            tau = -a / b;
            s = 1 / sqrt(1 + tau * tau); 
            c = s * tau;
        }   
        else
        {   
            tau = -b / a;
            c = 1 / sqrt(1 + tau * tau);
            s = c * tau;
        }
    }
}

void SVM::reduceCholFactor(const Array<double, 1>& T, const int idx)
{

	//
	//   Here, we downdate the Cholesky factorization by removing the 
	//   row/column, indexed by idx. There are two cases to consider. 
	//   (1) if 2 <= idx <= end, remove the row, perform Givens rotations, and 
	//   convert back to upper triangular and return the reduced R
	//   (2) if idx = 1, apply the transformation A to R, then convert to upper
	//   triangular and reduced the R to the new size. In this case, the
	//   transform A is defined as 
	//       [-y2 * [y3, ..., yn] 1; 
	//               I            0];
	//   

    int N = m_R.extent(0);

	if (N > 1)
	{
		if (idx > 0)
		{
		   // The cholesky reduction works by deleting the appropriate
		   // column of R and, then, performing a series of Givens
		   // rotations to convert the Hessenberg matrix back to 
		   // a upper right triangular matrix. That is, deleting 
		   // a column will introduce non-zero entries below the
		   // diagonal and using a givens rotation to zero out these
		   // entries will correct factorization.


		   // Shift the columns to the left, starting at idx, effectively
		   // deleting the column
		   for (int j = idx - 1; j < N - 1; j++)
		   {
			   for (int i = 0; i <= j + 1; i++)
			   {
				   m_R(i, j) = m_R(i, j + 1);
			   }
		   }


		   // Now we apply Givens rotations to zero out entries below the 
		   // diagonal.
		   for (int i = idx - 1; i < N - 1; i++)
		   {
			   double c = 0.0;
			   double s = 0.0;

			   // Compute the Givens rotation
			   computeGivens(m_R(i, i), m_R(i + 1, i), c, s);
		        
			   // Apply the Givens rotation
			   for (int k = i; k < N - 1; k++)
			   {
					double tau1 = m_R(i, k);
					double tau2 = m_R(i + 1, k);
					m_R(i, k) = c * tau1 - s * tau2;
					m_R(i + 1, k) = s * tau1 + c * tau2;
			   }
		   }    
		}
		else
		{

			// This first part computes the product of R*A. The product can be 
			// computed simply by noting that the first row is computed by adding
			// two elements of each column of R and the remaining rows consist
			// merely of a left shift of the original rows in R.
			//
			// Compute first row of m_R, in place. R(1,1) is the only element
			// that needs to be stored for future reference. As an alternative
			// to storing, the computation can be performed in reverse.
			//
			double R11 = m_R(0,0);
			for (int j = 0; j < N - 1; j++)
			{
				m_R(0,j) = -T(m_idxnb[1])*T(m_idxnb[j+2]) * R11 + m_R(0,j+1);
			}
	        
			// Now shift all of the entries to the left, below the first row.
			for (int i = 1; i < N; i++)
			{
				for (int j = 0; j < N - 1; j++)
				{
					m_R(i,j) = m_R(i, j+1);
				}
			}
	        
		   // Perform a series of Givens rotations on the entire matrix
		   // to convert the upper Hessenberg back to right triangular.

		   for (int i = 0; i < N - 1; i++)
		   {
			   double c = 0.0;
			   double s = 0.0;

			   // Compute the Givens rotation
			   computeGivens(m_R(i, i), m_R(i + 1, i), c, s);
		        
			   // Apply the Givens rotation
			   for (int k = i; k < N - 1; k++)
			   {
					double tau1 = m_R(i, k);
					double tau2 = m_R(i + 1, k);
					m_R(i, k) = c * tau1 - s * tau2;
					m_R(i + 1, k) = s * tau1 + c * tau2;
			   }
		   }    
		}    


		// Resize the factorization
		m_R.reference(m_RStorage(Range(0, N-2), Range(0, N-2)));
	}
	else
	{
		m_R.reference(m_RStorage(Range(0,0), Range(0,0)));
		m_R(0,0) = 0;
	}

    END_PROF();
}


void SVM::addToCholFactor(const Array<double, 1>& T, const int idx)
{
    
	//  
	//   Q, here, is the portion of the larger Q for the current non-bound support
	//   vectors. The last row/column represents the row/column to be added. T
	//   is the set of labels for the non-bound support vectors and the last
	//   entry represents the entry to be added. 
	// 
	//  Update the Cholesky factorization by solving the following
    //
	//  R^T*r = -y_1 * y_n * Z^T * Q * e_1 + Z^T * q
	//  r^T*r + rho^2 = e_1^T * Q * e_1 - 2 * y_1 * y_n * e_1^T * q + sigma
    //
	//  Note that Z is dealt with implicitly, i.e., no need to form due to sparse
    //  nature of the matrix.
	//
 
	int N = (int)m_idxnb.size();
    
    if (m_RStorage.extent(0) < N + 1 )
    {
        m_RStorage.resizeAndPreserve(N + 1 + STORAGE_NEW_INCREMENT, 
                                     N + 1 + STORAGE_NEW_INCREMENT);
    }

	if (N > 0)
	{
		// Reference a matrix which will be size N - 1 after adding 
		// the new entry
		m_R.reference(m_RStorage(Range(0, N - 1), Range(0, N - 1)));

		// Zero out new row/column
		for (int i = 0; i < N - 1; i++)
			m_R(i,N-1) = m_R(N-1,i) = 0;
		m_R(N-1,N-1) = 0;


		if (N == 1)
		{	
			m_R(0, 0) = sqrt(m_kernel.getQss(m_idxnb[0]) + m_kernel.getQss(idx) - 
				2 * T(m_idxnb[0]) * T(idx) * m_kernel(m_idxnb[0], idx));
		}
		else
		{
	                     
			//
			// Solve the following system
			// Z = [-T(1) * T(2:end-1); eye(length(T) - 2) ] ;
			// r = m_R' \(-T(1)*T(end) * Z' * Q(1:end-1,1) + Z' * q) ;
			//
			Array<double, 1> q(N - 1);

			// Form the RHS. Note that we are explicitly forming -rhs.
			for (int i = 0; i < N - 1; i++)
			{
			   q(i) = -T(m_idxnb[0]) * T(m_idxnb[i+1]) * 
						(m_kernel.getQss(m_idxnb[0]) * -T(m_idxnb[0]) * T(idx) + m_kernel[m_idxnb[0]][idx]) + 
						m_kernel[m_idxnb[i+1]][m_idxnb[0]] * -T(m_idxnb[0]) * T(idx) + 
						m_kernel[m_idxnb[i+1]][idx];
			}

	       
			//
			// Now solve the system through a forward substitution of 
			// triangular system
			// R' = x 0 0
			//      x x 0
			//      x x x
			//
			for (int i = 0; i < N - 1; i++)
			{
				m_R(i, N - 1) = q(i);
				for (int j = 0; j <= i - 1; j++)
					m_R(i, N - 1) -= m_R(j,i) * m_R(j, N - 1);
				m_R(i, N - 1) /= m_R(i, i);
			}

			// Solve for rho
			double sigma = 0;
			for (int i = 0; i < N - 1; i++)
				sigma += m_R(i, N - 1) * m_R(i, N - 1);

			m_R(N - 1,N - 1) = sqrt(m_kernel.getQss(m_idxnb[0]) - 2 * T(m_idxnb[0]) * T(idx) * 
									m_kernel[m_idxnb[0]][idx] + m_kernel.getQss(idx) - sigma);

		}
	}
}

