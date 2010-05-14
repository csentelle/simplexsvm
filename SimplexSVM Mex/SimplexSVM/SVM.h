#pragma once

#include "Kernel.h"
#include "ProxyStream.h"	


#include <blitz/array.h>         // include Blitz header file


BZ_USING_NAMESPACE(blitz) 

#include <vector>
using namespace std;

class SVM
{
public:

    SVM(KernelCache& kernel, double C, double tol, ProxyStream& os);
    virtual ~SVM(void);

    void train(const Array<double, 2>& P, 
               const Array<double, 1>& T,
               Array<double, 1>& alpha,
               double& b,
               int& iter,
               double& t,
               int working_size,
               int shrink_iter); 

private:

    Array<double, 1>& updateCache(Array<double, 1>& fcache,
                                  const Array<double, 1>& T,
								  const Array<double, 1>& alpha,
								  const double beta,
                                  const Array<double, 1>& alphaOld,
                                  const double betaOld,
                                  const int pivotIdx);

    void reinitializeCache(Array<double, 1>& fcache,
                           Array<double, 1>& upperfcache,
                            const Array<double, 1>& T,
						    const Array<double, 1>& alpha,
						    const double beta);


    // Add a row/column to the factorization (L)
    void addToCholFactor(const Array<double, 1>& T, const int idx, const int iter);
    void reduceCholFactor(const Array<double, 1>& T, const int idx);

    void solveSubProblem(const Array<double, 2>& R, 
                         const Array<double, 1>& qs,
                         const double ys,
                         const Array<double, 1>& T,
                         Array<double, 1>& h,
                         double& g);

    void takeStep(Array<double, 1>& alpha, 
                  const int idx,
                  Array<double, 1>& fcache,
                  const Array<double, 1>& T,
                  double& beta,
                  Array<double, 1>& uppperfcache,
				  int& iter);


    int updateCacheStrategy(const int nWorkingSize, 
                             const Array<double, 1>& T,
                             const Array<double, 1>& alpha, 
                             const double beta,
                             Array<double, 1>& fcache,
                             Array<double, 1>& upperfcache,
                             double& ming);

	inline Array<double, 1>& fwrdSolve(const Array<double, 2>& R, Array<double, 1>& x);
	inline Array<double, 1>& bkwrdSolve(const Array<double, 2>& R, Array<double, 1>& x);

    KernelCache& m_kernel;
    ProxyStream& m_os;
    const double m_C;
	const double m_tol;

    vector<int> m_idxnb;
    vector<int> m_idxb;
    vector<int> m_changed;
    vector<int> m_workset;
    vector<int> m_fcache_indices;
    vector<int> m_activeset;
    vector<int> m_idxpossible;
    vector<int> m_idxreplace;
    Array<int, 1> m_status;

    Array<double, 2> m_R;


	//
	// These represent arrays that can be preallocated and referenced by the 
	// actual arrays that grow/shrink in size. This addresses a limitation in
	// the Blitz arrays where dynamic memory allocation would occur each time
	// shrinking or growing occurs
	//
    Array<double, 2> m_RStorage;
    Array<double, 1> m_hS;
    Array<double, 1> m_etS;
    Array<double, 1> m_qS;

	// Temporary buffer for computing the error caches
	double* m_kernelCacheBuffer;

	//
	// The m_RStorage array is not preallocated to the maximum size since
	// storage is O(N^2) spatial complexity. Instead, m_RStorage is preallocated
	// in chunks, similar to the manner in which the STL library would implement 
	// this. The following defines how many entries should be added each time
	// a dynamic allocation occurs.
	//
    const static int STORAGE_NEW_INCREMENT = 100;
};
