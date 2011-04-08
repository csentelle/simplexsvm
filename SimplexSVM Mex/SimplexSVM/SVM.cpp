
#pragma warning (push)
#pragma warning( disable : 4805)
#pragma warning( disable : 4800)


#include ".\svm.h"
#include <algorithm>
#include <functional>
#include "time.h"
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
        return m_arr((int)idx1) < m_arr((int)idx2);
    }

    const T& m_arr;
};

template <typename T> 
T sign(T t)  
{ 
    if( t == 0 ) 
        return T(0); 
    else 
        return (t < 0) ? T(-1) : T(1); 
} 

template <typename T>
T max(T& a, T& b)
{
	return a > b : a ? b;
}

double vmaxabs(Array<double,1>& a, vector<int>& idx)
{
	double maxval = 0.0;
	for (int i = 0; i < idx.size(); i++)
		maxval = max(maxval, abs(a(idx[i])));

	return maxval;

}

SVM::SVM(KernelCache<double, float>& kernel, double C, double tol, ProxyStream& os)
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
	delete [] m_kernelCacheBuffer;
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

	m_kernelCacheBuffer = (double*) malloc(T.size() * sizeof(double));
	if (!m_kernelCacheBuffer)
		throw exception("Error allocating kernel cache buffer");

    //
    // Allocate temporary arrays, these will be used in some functions
    // to prevent heap allocation multiple times, for efficiency.
    //

    m_hS.resize(fcache.shape());
    m_etS.resize(fcache.shape());
    m_qS.resize(fcache.shape());
	
    m_RStorage.resize(STORAGE_NEW_INCREMENT, STORAGE_NEW_INCREMENT);
    m_R.reference(m_RStorage(Range(0,0), Range(0,0)));

	m_gp.resize(fcache.shape());

    m_status.resize(fcache.shape());

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

	m_shrink.assign(alpha.size(), 1);

    reinitializeCache(fcache, upperfcache, T, alpha, beta);

    m_R(0, 0) = sqrt(m_kernel.getQss(initS));

	//
    // Find the next index to enter the cache. Note that the 
    // minIndex function returns a TinyVector, of which we 
    // want the first value.
    //
    int idx = minIndex(fcache)(0);
    double min_g = min(fcache);


    iter = 0;
    time_t ts = clock();
	Profile.reset();

	int cycles = 0;
	while (min_g < -m_tol )

    {
               		
        m_os << infolevel(1) << "iter = " << iter 
                             << " min_g = " << min_g 
							 << " beta = " << beta
                             << " idx = " << idx << endl;

		m_os << infolevel(7) << "fcache = " << fcache << endl;

		BEGIN_PROF("takeStep");
		
		if ((++cycles % 2) == 0)
		{
			gradientProjection(alpha, 
							   fcache,
							   T,
							   beta, 
							   upperfcache);
		}
		else
		{
			// Perform pivoting on the pivot element
			takeStep(alpha, idx, fcache, T, beta, upperfcache, iter);
		}

		NEXT_PROF("Update Cache Strategy");

		idx = updateCacheStrategy(working_size, 
								  T, 
								  alpha, 
								  beta, 
								  fcache, 
								  upperfcache, 
								  min_g);
        
		END_PROF();
		
		// 	If we appear to have hit the stopping criterion, then, proceed 
		//  first with a reinitialization of a working set corresponding to the 
		//  partial pricing technique. If this doesn't work, then we need to 
		//  recompute the entire error cache for sprinting purposes. m_workset
		//  is the set of indices currently being worked on by the partial 
		//  pricing technique and m_fcache_indices are the current set of indices
		//  being worked on for shrinking.
        if (min_g > -m_tol)
        {
			BEGIN_PROF("Reset working set");
            
            m_workset.erase(m_workset.begin(), m_workset.end());
            
            idx = updateCacheStrategy(working_size, 
                                      T, 
                                      alpha, 
                                      beta, 
									  fcache, 
									  upperfcache, 
									  min_g);

			END_PROF();
            
			if (min_g > -m_tol)
            {
				BEGIN_PROF("Reset shrinking");
				// Reinitialize the active set. Note that we need to rename
				// m_fcache_indices to activeIndices since this is what it really
				// is.
                m_fcache_indices.erase(m_fcache_indices.begin(),
                                       m_fcache_indices.end());
				
                for (int i = 0; i < fcache.size(); i++)
                    m_fcache_indices.push_back(i);

				// Erase the working set, which will force a new working set computation
				// on the entire data set.
				m_workset.erase(m_workset.begin(), m_workset.end());
                
				// Reset kernel caching. This will force recomputation of entire columns
				//m_kernel.resetActiveToFull();

				// reinitialize the upper cache
				//reinitializeUpperCache(upperfcache);

                idx = updateCacheStrategy(working_size, 
                                          T, 
                                          alpha, 
                                          beta, 
										  fcache, 
										  upperfcache, 
										  min_g);
				END_PROF();

            }
        }

        // Perform shrinking
        else if (iter % shrinking_iter == 0)
        {
			
			BEGIN_PROF("Shrinking");
			// Make sure all entries have been updated, not just the working set
            reinitializeCache(fcache, upperfcache, T, alpha, beta);

            for (int i = 0; i < m_fcache_indices.size(); i++)
            {
                //if (fcache(m_fcache_indices[i]) > m_tol)
				
				// Double checking that fcache > 0 prevents shrinking of non-bound
				// SVs as well other data points that haven't been part of the
				// working set from being excluded.
                if (m_shrink[m_fcache_indices[i]] && fcache(m_fcache_indices[i]) > m_tol)
				{

					//m_kernel.removeActiveIndex(m_fcache_indices[i]);

                    m_fcache_indices.erase(remove(m_fcache_indices.begin(),
                                                  m_fcache_indices.end(),
                                                  m_fcache_indices[i]),
                                           m_fcache_indices.end());

                }
            }

			m_shrink.assign(alpha.size(),1);

			// Erase the working set, which will force a new working set computation
			// on the entire data set.
			m_workset.erase(m_workset.begin(), m_workset.end());            

            idx = updateCacheStrategy(working_size, 
                                      T, 
                                      alpha, 
                                      beta, 
									  fcache, 
									  upperfcache, 
									  min_g);
			END_PROF();    
		}
	
    }


	//Profile.dumpprint(m_os, 0);
    t = ((double)clock() - (double)ts) / (double)CLOCKS_PER_SEC;

    // Compute B
    // We are computing B according to:    
    b = beta;

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


	BEGIN_PROF("H");
	// Force caching of the column
	//m_kernel.updateColumn(m_idxnb[i]);
 
	NEXT_PROF("H2");
    for (int i = 0; i < q.size(); i++)
    {
		q(i) = m_kernel.getUnsafeCachedItem(m_idxnb[i], idx);
    }
	END_PROF();
    m_os << infolevel(4) << "q = " << q << endl;

    // First, we are adding a new variable to the basis, identified by idx    
    // We are adding a support vector, alpha is increased from zero. 

    m_os << infolevel(4) << "m_status = " << m_status(idx) << endl;

	BEGIN_PROF("TAKE STEP 1");
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
    
	NEXT_PROF("TAKE STEP 2");
    bool bAddedIndex = false;
                
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

				const float* buffer = m_kernel.getColumnAndLock(idxr);

				for (int k = 0; k < upperfcache.size(); k++)
                    upperfcache(k) += buffer[k] * m_C;

				m_kernel.unlockColumn(idxr);

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

				m_os << infolevel(20) << "Reduced Cholesky = " << endl;
				m_os << infolevel(20) << m_R << endl;            

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


				const float* buffer = m_kernel.getColumnAndLock(idx);

                // Update the upper bound cache
				for (int k = 0; k < upperfcache.size(); k++)
                    upperfcache(k) -= buffer[k] * m_C;

				m_kernel.unlockColumn(idx);
            }

            m_os << " idx = " << idx << " idxr = " << idxr << endl;

            if (idx != idxr)
            {

                // Update the cholesky factorization
                addToCholFactor(T, idx, iter);

				//
                // Go ahead and add to the non-bound support vector list. This has to be done after adding 
                // to the factorization, otherwise, the factorization will be incorrect.
                //
                m_idxnb.push_back(idx);
                m_status(idx) = 1;
                    
                m_os << infolevel(20) << "New Chol Factorization = " << endl << m_R << endl;
            }
        
            bAddedIndex = true;
        }


		//
        // Now we need to drive the corresponding Lagrange to zero to force
        // the complementary conditions
        //
        if (abs(fcache(idx)) <= m_tol) 
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
	END_PROF();
}

void SVM::gradientProjection(Array<double, 1>& alpha, 
						     Array<double, 1>& fcache,
							 const Array<double, 1>& T,
							 double& beta, 
							 Array<double, 1>& upperfcache)
{

	vector<int> I(fcache.size());
	vector<int> d(fcache.size());

	for (int i = 0; i < T.size(); i++)
        I[i] = i;
	
    //
    // Here, we borrow a methodology similar to that presented by Joachims.
    // The steps are as follows:
    //
    for (int idx = 0; idx < fcache.size(); idx++)
	{
		m_gp(idx) = fcache(idx) * T(idx);
		d[idx] = (int)sign(fcache(idx));
	}

    //
    // Note that d = -1 for alpha = 0 and KKT violator
    //           d =  1 for alpha = C and KKT violator
    //
	for (size_t idx = 0; idx < m_idxb.size(); idx++)
	{
		m_gp(m_idxb[idx]) *= -1;
		d[m_idxb[idx]] *= -1;
	}
    
	// Perform an ascending sort. The m_fcache_indices contain our data.
	sort(I.begin(), 	     
	     I.end(), 
		 IndexCompare<Array<double,1> >(m_gp));


    const int Ns = 500;    
    
	//
    // Take pairs from the top/bottom such that d < 1 for alpha = 0 and d >
    // 1 for alpha = C. We will search for no more than Ns pairs of points.
    //     
    size_t idxtop = 0;
    size_t idxbottom = m_fcache_indices.size() - 1;
    int idxfound = 0;

	vector<double> adj(fcache.size());
	vector<double> a1(fcache.size());

    while (idxfound < Ns)
	{               
        while (((d[I[idxtop]] == 1 && m_status(I[idxtop]) == 0) || 
               (d[I[idxtop]] == -1 && m_status(I[idxtop]) == 2) ||
			   (m_status(I[idxtop]) == 1)) &&
               m_gp(I[idxtop]) < 0)
		{
            idxtop = idxtop + 1;                       
		}

        if (m_gp(I[idxtop]) >= 0) 
			break; 
        
        while (((d[I[idxbottom]] == 1 && m_status(I[idxbottom]) == 0) || 
               (d[I[idxbottom]] == -1 && m_status(I[idxbottom]) == 2) ||
			   (m_status(I[idxtop]) == 1)) && 
                m_gp(I[idxbottom]) > 0)
		{
            idxbottom = idxbottom - 1;                       
		}

        if (m_gp(I[idxbottom]) <= 0) 
			break; 
        
        // Determine if the pair will create a strictly decreasing objective
        // idxs = [idxtop; idxbottom];
        
		vector<size_t> idxs(2);
		idxs[0] = idxtop;
		idxs[1] = idxbottom;

        //
        // Test to ensure the pair of points will provide a strict decrease
        // in the objective function. The formula is as follows
        // 
        // d'*(Q_rs*alpha_s + Q_rc * alpha_c - 1_r - C*Q_rr*d) >= 0
        //
        // We observe that we can modify the formulas as follows:
        //
        // d'*(-d.*m_fcache + beta*y_r - C*Q_ss*d) >= 0 
        // 
        // where d_i = -1 if alpha_i = 0, d_i = 1 if alpha_i = C

        //
        // Note that we use -d based upon the definition applied, above.
        //
		double val = 0.0;
		for (size_t k = 0; k < idxs.size(); ++k)
		{
			val += d[I[idxs[k]]] * (-d[I[idxs[k]]] * (fcache(I[idxs[k]]) + adj[I[idxs[k]]]) + beta * T(I[idxs[k]]) - 
					0.5 * m_C * (m_kernel(I[idxs[k]], I[idxs[0]]) * d[I[idxs[0]]] + 
					             m_kernel(I[idxs[k]], I[idxs[1]]) * d[I[idxs[1]]]));

		}
        
        // Update the alpha value and fcache values  
        if (val > 0)
		{
            
            idxfound = idxfound + 1;
			for (size_t k = 0; k < idxs.size(); ++k)
			{
				
				if (m_status(I[idxs[k]]) == 2)
				{
					// Update variable type vectors
					m_idxb.erase(remove(m_idxb.begin(), 
										m_idxb.end(), 
										I[idxs[k]]), 
								 m_idxb.end());
				}
				else
				{
					m_idxb.push_back(I[idxs[k]]);
				}

				alpha(I[idxs[k]]) = m_C - alpha(I[idxs[k]]);
				m_status(I[idxs[k]]) = 2 - m_status(I[idxs[k]]);
			}

			// Compute an update to the kernel caching
            // adj = adj - Q(:,I(idxs)) * (-d(I(idxs)) * m_C);


			for (int i = 0; i < a1.size(); i++)
			{				
				//a1 = Q(:,I(idxs)) * (-d(I(idxs)) * m_C);
				for (size_t k = 0; k < idxs.size(); k++)
				{
					a1[i] = m_kernel(i, I[idxs[k]]) * -d[I[idxs[k]]] * m_C;

				}

                // a1(m_svtype==2) = -a1(m_svtype==2);
				if (m_status(i) == 2)
					a1[i] = -a1[i];

		
			}

			for (int k = 0; k < idxs.size(); k++)
			{

				//a1(I(idxs)) = ((-2*(m_fcache(I(idxs))'+adj(I(idxs)))) + a1(I(idxs)));
				a1[I[idxs[k]]] += -2 * (fcache(I[idxs[k]]) + adj[I[idxs[k]]]);
			}

			for (int i = 0; i < adj.size(); i++)
			{
				adj[i] += a1[i];
			}

		}

        idxtop = idxtop + 1;
        idxbottom = idxbottom - 1;        
	}       

	// Apply adjustments to upperfcache as well as fcache
	for (int i = 0; i < fcache.size(); i++)
	{
		fcache(i) += adj[i];
		upperfcache(i) += adj[i];
	}

	if (idxfound > 0)
	{
		double eps = 1e-15;
		int N = T.size();

		Array<double, 1> h = m_hS(Range(0, (int)m_idxnb.size() - 1)); 
		double gb = 0.0;

		for (int kidx = 0; kidx < h.size(); kidx++)
			h(kidx) = 0.0;

		// Find the indices for bound, non-bound support vectors. Note that 
		// this, in reality, needs to be moved to another section.
		Array<double, 1> rhs = m_qS(Range(0, (int)m_idxnb.size() - 1));

		for (int i = 0; i < rhs.size(); i++)
		{
			rhs(i) = fcache(m_idxnb[i]);
		}

		m_os << infolevel(4) << "rhs = " << rhs << endl;

		//
		// Solve sub-problem
		//
		solveSubProblem(m_R, 
						rhs,
						0,
						T,
						h,
						gb);

	        

		m_os << infolevel(4) << "Calculations for g, h" << endl;
		m_os << infolevel(4) << "gb = " << gb << endl;
		m_os << infolevel(4) << "ht = " << h << endl;


        //
        // Compute gamma. Note that we don't actually need the term
        // for g(idx + 1) since this will be zero as idx represents
        // a non-support vector and g is only non-zero for bound 
        // support vectors (m_status == 2)
        //
		vector<double> gamma(m_idxnb.size());

		//gamma = -Q(idx_nb,idx_nb) * h(idx_nb)' - gb * m_T(idx_nb)';
		for (int i = 0; i < m_idxnb.size(); i++)
		{
			gamma[i] = -gb * T(m_idxnb[i]);
			for (int j = 0; j < m_idxnb.size(); j++)
			{
				gamma[i] += m_kernel(m_idxnb[i], m_idxnb[j]) * h(j);
			}
		}
        
                  		
		
		while (vmaxabs(fcache, m_idxnb) > m_tol)
		{
			// 
			// Search for the minimum theta. 
			// Augment the h vector for non-bound support vectors        
			//
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
				idxt = m_idxnb[i];

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

			double ratiomax = 0.0;
			for (int i = 0; i < m_idxnb.size(); i++)
				if (fcache(m_idxnb[i])/gamma[i] > ratiomax)
					ratiomax = fcache(m_idxnb[i])/gamma[i];



			// Note that we cannot use the "tol", here, for this decision, but a much tighter
			// eps, otherwise cycling can occur. 
			if ( (theta < ratiomax - eps))
			{

				m_os << infolevel(2) << "Value leaving basis: " << idxr << endl;

				// a value is leaving the basis
				beta = beta - theta * gb;

				for (int k = 0; k < h.size(); k++)
				{
					alpha(m_idxnb[k]) = alpha(m_idxnb[k]) - theta * h(k);
					fcache(m_idxnb[k]) = fcache(m_idxnb[k]) - theta * gamma[k];
				}

	               
				m_os << infolevel(7) << "beta = " << beta << endl;
				m_os << infolevel(7) << "alpha = " << alpha << endl;

				// If we added to the bound set, that is we hit alpha = C.
				if (-h(idxh) > 0)
				{

					m_idxb.push_back(idxr);
					m_status(idxr) = 2;

					const float* buffer = m_kernel.getColumnAndLock(idxr);

					for (int k = 0; k < upperfcache.size(); k++)
						upperfcache(k) += buffer[k] * m_C;

					m_kernel.unlockColumn(idxr);

				}
				else
				{
					m_status(idxr) = 0;
				}


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

				m_os << infolevel(20) << "Reduced Cholesky = " << endl;
				m_os << infolevel(20) << m_R << endl;            

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
			else
			{
		
				idxr = -1;
				m_os << infolevel(2) << "Found minimum of objective." << endl;
	           
				theta = 1.0;
	            
				// a value is leaving the basis
				beta = beta - theta * gb;

				for (int k = 0; k < h.size(); k++)
				{
					alpha(m_idxnb[k]) = alpha(m_idxnb[k]) - theta * h(k);
					fcache(m_idxnb[k]) = fcache(m_idxnb[k]) - gamma[k];
				}
	            

				m_os << infolevel(7) << "beta = " << beta << endl;
				m_os << infolevel(7) << "alpha = " << alpha << endl;
	           
			}           
	  
			//
			// Now we need to drive the corresponding Lagrange to zero to force
			// the complementary conditions
			//
			if (vmaxabs(fcache, m_idxnb) <= m_tol) 
				break; 
	        
			h.reference(m_hS(Range(0, (int)m_idxnb.size() - 1)));
	    
			m_os << infolevel(2) << "~bSlack" << endl;           
	        
			// Solve sub-problem
			Array<double, 1> et = m_etS(Range(0, h.size() - 1));

			for (int i = 0; i < m_idxnb.size(); i++)
			{
				et(i) = fcache(m_idxnb[i]);
			}

			m_os << infolevel(4) << "et = " << et << endl;
	       
			solveSubProblem(m_R, 
							et,
							0,
							T,
							h,
							gb);
	        
			m_os << infolevel(4) << "gb = " << gb << endl;
			m_os << infolevel(4) << "h = " << h << endl;
	                    
			//gamma = -Q(idx_nb,idx_nb) * h(idx_nb)' - gb * m_T(idx_nb)';
			for (int i = 0; i < m_idxnb.size(); i++)
			{
				gamma[i] = -gb * T(m_idxnb[i]);
				for (int j = 0; j < m_idxnb.size(); j++)
				{
					gamma[i] += m_kernel(m_idxnb[i], m_idxnb[j]) * h(j);
				}
			}


		} // while


	}

}


int SVM::updateCacheStrategy(const int nWorkingSize, 
                             const Array<double, 1>& T,
                             const Array<double, 1>& alpha, 
                             const double beta,
                             Array<double, 1>& fcache,
                             Array<double, 1>& upperfcache,
                             double& ming)
{

	BEGIN_PROF("Update Cache Strategy");

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
		vector<const float*> vctr(m_idxnb.size());

		for (size_t k = 0; k < m_idxnb.size(); k++)
			vctr[k] = m_kernel.getColumnAndLock(m_idxnb[k]);

		for (int i = 0; i < m_workset.size(); i++)
		{
			int idx = m_workset[i];

			if (m_status(idx) == 0)
			{
				fcache(idx) = -1 - T(idx) * beta + upperfcache(idx);
				for (int k = 0; k < m_idxnb.size(); k++)
					fcache(idx) += vctr[k][idx] * alpha(m_idxnb[k]);

			}
			else if (m_status(idx) == 2)
			{
				fcache(idx) =  1 + T(idx) * beta - upperfcache(idx);
				for (int k = 0; k < m_idxnb.size(); k++)
					fcache(idx) -= vctr[k][idx] * alpha(m_idxnb[k]);

			}
			else
			{
				fcache(idx) = 0.0;
			}
		}
		m_kernel.unlockAllColumns();
	}

    // Find the minimum in the workset.
    int idxmin = 0;
    ming = 1e300;

	for (int i = 0; i < m_workset.size(); i++)
    {
		m_shrink[m_workset[i]] = fcache(m_workset[i]) > 0 && m_shrink[m_workset[i]];

        if (fcache(m_workset[i]) < ming)
        {
            ming = fcache(m_workset[i]);
            idxmin = m_workset[i];
        }
    }

	END_PROF();
    return idxmin;
	
}

void SVM::reinitializeUpperCache(Array<double, 1>& upperfcache)
{
	BEGIN_PROF("reinitializeUpperCache");

	for (int i = 0; i < m_fcache_indices.size(); i++)
		upperfcache(m_fcache_indices[i]) = 0;

	for (int i = 0; i < m_idxb.size(); i++)
	{
		int idx = m_idxb[i];
		m_kernel.updateColumn(idx);

		// Update the upper bound cache
		for (int k = 0; k < m_fcache_indices.size(); k++)
			upperfcache(m_fcache_indices[k]) += m_kernel.getUnsafeCachedItem(idx, m_fcache_indices[k]) * m_C;
	}
	END_PROF();
}


void SVM::reinitializeCache(Array<double, 1>& fcache,
                            Array<double, 1>& upperfcache,
                            const Array<double, 1>& T,
							const Array<double, 1>& alpha,
							const double beta)
{

	BEGIN_PROF("Update columns");
	vector<const float*> vctr(m_idxnb.size());
	for (size_t k = 0; k < m_idxnb.size(); k++)
		vctr[k] = m_kernel.getColumnAndLock(m_idxnb[k]);

	NEXT_PROF("Compute errors");
    for (int i = 0; i < m_fcache_indices.size(); i++)
	{
        int idx = m_fcache_indices[i];

		if (m_status(idx) == 0)
        {
			fcache(idx) = -1 - T(idx) * beta + upperfcache(idx);
			for (int k = 0; k < m_idxnb.size(); k++)
			{
				fcache(idx) += vctr[k][idx] * alpha(m_idxnb[k]);
			}

        }
        else if (m_status(idx) == 2)
        {
            fcache(idx) =  1 + T(idx) * beta - upperfcache(idx);
			for (int k = 0; k < m_idxnb.size(); k++)
			{
				fcache(idx) -= vctr[k][idx] * alpha(m_idxnb[k]);
			}

		}
		else
        {
			fcache(idx) = 0.0;
        }
	}
	m_kernel.unlockAllColumns();
	END_PROF();

}

void SVM::solveSubProblem(const Array<double, 2>& R, 
                          const Array<double, 1>& qs,
                          const double ys,
                          const Array<double, 1>& T,
                          Array<double, 1>& h,
                          double& g)
{

    BEGIN_PROF("solveSubProblem");
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
		m_kernel.updateColumn(m_idxnb[0]);

		for (int i = 0; i < N - 1; i++)
		{
			assert(i >= 0 && i < N - 1);
            hz(i) = T(m_idxnb[0]) * T(m_idxnb[i + 1]) * (m_kernel.getQss(m_idxnb[0]) * hy - qs(0)) +
				     qs(i + 1) - m_kernel.getUnsafeCachedItem(m_idxnb[0], m_idxnb[i + 1]) * hy;
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
			g += T(m_idxnb[0]) * m_kernel.getUnsafeCachedItem(m_idxnb[0], m_idxnb[i]) * h(i);
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

	END_PROF();
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
	BEGIN_PROF("reduceCholFactor");

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


void SVM::addToCholFactor(const Array<double, 1>& T, const int idx, const int iter)
{
    BEGIN_PROF("addToCholFactor");
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

			m_kernel.updateColumn(m_idxnb[0]);
			
			// Form the RHS. Note that we are explicitly forming -rhs.
			for (int i = 0; i < N - 1; i++)
			{
			   q(i) = -T(m_idxnb[0]) * T(m_idxnb[i+1]) * 
						(m_kernel.getQss(m_idxnb[0]) * -T(m_idxnb[0]) * T(idx) + m_kernel.getUnsafeCachedItem(m_idxnb[0],idx)) + 
						m_kernel.getUnsafeCachedItem(m_idxnb[0], m_idxnb[i+1]) * -T(m_idxnb[0]) * T(idx) + 
						m_kernel.getUnsafeCachedItem(m_idxnb[i+1], idx);

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
									m_kernel.getUnsafeCachedItem(m_idxnb[0],idx) + m_kernel.getQss(idx) - sigma);


		}
	}
	END_PROF();
}

#pragma warning (pop)