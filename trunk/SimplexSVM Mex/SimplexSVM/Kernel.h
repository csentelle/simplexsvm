#pragma once

// For Blitz Array class.
#include <blitz/array.h>         
BZ_USING_NAMESPACE(blitz) 
#include <list>
#include <vector>
using namespace std;

template<class T>
inline void INSERT_ELEMENTS(T& coll, int first, int last)
{
	for (int i = first; i <= last; ++i)
		coll.insert(coll.end(),i);
};

template<class T>
void swapdata(T& a, T& b)
{
	T tmp = a;
	a = b;
	b = tmp;
};

struct SparseNode
{
	SparseNode(double value, int index) 
		: m_value(value), m_index(index) {}

	double m_value;
	int m_index;
};

// CacheLine
// This structure stores information regarding a cache line. A cacheline is a column 
// of the kernel matrix. The length of the cache line is dependent upon the current
// active size, variables that have not been removed through the shrinking process.
//
struct CacheLine
{
	// Default constructor
	CacheLine() 
		: m_buffer(NULL), m_len(0), m_index(0) {};

	CacheLine(double* buffer, size_t len, size_t index) 
		: m_buffer(buffer), m_len(len), m_index(index) {};

	double* m_buffer;
	size_t m_len;	
	size_t m_index;
};


class Kernel
{
public: 
	Kernel() {}
	virtual ~Kernel() {}

	virtual double compute(const Array<double, 2>& P, const Array<double,1>& T, 
		const size_t i, const size_t j) const = 0;
};

class LinearKernel : public Kernel
{
public:

	LinearKernel() : Kernel() {};
	virtual ~LinearKernel(void) {};

	double compute(const Array<double, 2>& P, const Array<double,1>& T, 
		const size_t i, const size_t j) const
	{
		size_t N = T.size();
		double sum = 0.0;

		const double *points1 = &P.data()[ i * N ];
		const double *points2 = &P.data()[ j * N ];

		for (size_t k = 0; k < N; k++)
			sum += points1[k] * points2[k];

		// Note weakness of Blitz++ arrays, which should be using size_t instead of int.
		return sum * T((int)i) * T((int)j);
	}
};


class RBFKernel: public Kernel
{
public:

	RBFKernel(double fpGamma): Kernel(), m_fpGamma(fpGamma) {};
	virtual ~RBFKernel(void) {};

	double compute(const Array<double, 2>& P, const Array<double, 1>& T, 
		const size_t i, const size_t j) const
	{
		size_t N = T.size();	
		double sum = 0.0;
		double diff = 0.0;

		const double* points1 = &P.data()[ i * N ];
		const double* points2 = &P.data()[ j * N ];

		for (size_t k = 0; k < N; k++)
		{
			diff = points1[k] - points2[k];
			sum += diff * diff;
		}

		return exp(-m_fpGamma * sum) * T((int)i) * T((int)j);
	}

private:

	const double m_fpGamma;

};


class KernelCache
{
public:

	KernelCache(const Kernel& kernel, 
		const Array<double, 2>& P, 
		const Array<double, 1>& T,
		unsigned int cacheSize = 100) 
		: m_kernel(kernel)
		, m_P(P)
		, m_T(T)
		, m_cacheSize(cacheSize * 1024 * 1024)
		, m_currentAllocSize(0)
		, m_colIndices(T.size())
		, m_rowIndices(T.size())
		, m_activeLength(T.size())
	{

		for (int i = 0; i < T.size(); i++)
		{
			m_cacheDiagonal[i] = kernel.compute(m_P, m_T, i, i);
		}

		// Initialize the active list and all of the row indices
		INSERT_ELEMENTS(m_activeIndices, 0, m_T.size() - 1);
		INSERT_ELEMENTS(m_rowIndices, 0, m_T.size() - 1);

		// Add a NULL reference. The item at the end of the list will contain a pointer to
		// an empty buffer. This is used as a NULL reference since iterators cannot be
		// specified as NULL.
		m_cache.push_back(CacheLine());
		NULLITEM = m_cache.begin();

	}


	virtual ~KernelCache(void) 
	{
		// Clear memory allocated by cache. Remaining objects should be cleaned up
		// by default destructors
		clearCache();
	}

	double operator()(size_t i, size_t j) 
	{
		assert(i < m_colIndices.size() && j < m_activeLength);
		return getColumn(i)[m_rowIndices[j]];
	}

	void getColumn(size_t idx, double* buffer, size_t& len)
	{
		double* colBuffer = getColumn(idx);

		len = min(len, (*m_colIndices[idx]).m_len);
		memcpy(buffer, colBuffer, len * sizeof(double));
	}

	const vector<size_t>& getActiveIndices() const
	{
		return m_activeIndices;
	}

	double getQss(size_t i)
	{
		return m_cacheDiagonal[i];
	}

	void resetActiveToFull()
	{
		// Reset the active indices length, which should subsequently signal
		// a need to recompute each of the cache lines as they are accessed
		m_activeLength = m_activeIndices.size();
	}

	void removeActiveIndex(size_t i)
	{
		// Swap this item out with the last element in the list.
		size_t idx = m_rowIndices[i];

		// Proceed through the items in each cache line and swap the computed value to be 
		// removed with the last in the active list
		for (list<CacheLine>::iterator pos = m_cache.begin(); pos != m_cache.end(); pos++)
		{
			swapdata((*pos).m_buffer[idx], (*pos).m_buffer[m_activeLength-1]);
			(*pos).m_len--;
		}

		// Swap entries in the active indices to place the removed variable at
		// the end of the current active list.
		swapdata(m_activeIndices[m_activeLength-1], m_activeIndices[idx]);

		// Swap the pointers to active indices in the indices structure
		swapdata(m_rowIndices[i], m_rowIndices[m_activeLength-1]);

		// Reduce the length of the active indices
		m_activeLength--;
	}

protected:

	double* getColumn(size_t i)
	{
		if ((*m_colIndices[i]).m_buffer != NULL)
		{
			refreshColumn(i);
		}
		else
		{
			cacheColumn(i);
		}

		return (*m_colIndices[i]).m_buffer;
	}

	// Proceed through the cache and clean up all cache lines
	void clearCache()
	{
		for (list<CacheLine>::iterator pos = m_cache.begin(); pos != m_cache.end(); ++pos)
		{
			delete [] (*pos).m_buffer;
			(*pos).m_len = 0;
			m_colIndices[(*pos).m_index] = NULLITEM;
		}		
	}


	void refreshColumn(size_t i)
	{

		if (m_colIndices[i] != NULLITEM)
		{
			// Move the item to the front of the list
			CacheLine cacheline = *m_colIndices[i];
			m_cache.erase(m_colIndices[i]);
			m_cache.push_front(cacheline);

			// Determine if the size is correct.
			if (cacheline.m_len < m_activeLength)
			{
				// Allocate a larger buffer
				reallocColumn(cacheline);

				// Really only need to compute a partial column for speed
				computeColumn(cacheline.m_buffer, i);
			}
		}		 		
	}

	void reallocColumn(CacheLine& cacheline)
	{
		if ((m_activeLength - cacheline.m_len) * sizeof(double) + m_currentAllocSize < m_cacheSize)
		{
			m_currentAllocSize -= cacheline.m_len * sizeof(double);

			// We need to resize the column
			cacheline.m_buffer = (double*)realloc(cacheline.m_buffer, m_activeLength * sizeof(double));
			cacheline.m_len = m_activeLength;

			m_currentAllocSize += m_activeLength * sizeof(double);				
		}
		else
		{
			throw "out of memory, increase cache size";
		}
	}

	void computeColumn(double* column, size_t i)
	{		
		// Compute the column
		for (size_t j = 0; j < m_activeIndices.size(); j++)
			column[j] = compute( i, m_activeIndices[j]);
	}

	// Cache a column of the kernel matrix. Only the portion corresponding
	// to the active indices is cached.
	void cacheColumn(size_t i) 
	{
		if (NULLITEM == m_colIndices[i])
		{
			size_t len = m_activeLength * sizeof(double);

			// If there is sufficient cache left over
			if (len + m_currentAllocSize < m_cacheSize)
			{
				// Just allocate a buffer.
				m_cache.push_front(CacheLine((double*)malloc(len), 
					m_activeLength, i));

				m_colIndices[i] = m_cache.begin();

				m_currentAllocSize += len;

				computeColumn((*m_colIndices[i]).m_buffer, i);

			}
			else
			{
				// Delete the LRU and realloc for the purposes of this item. Note
				// that the very last item is 
				list<CacheLine>::iterator pos = m_cache.end()-- --;

				m_colIndices[(*pos).m_index] = NULLITEM;
				(*pos).m_index = i;
				m_colIndices[i] = pos;

				// This will force a reallocation and recompute on the column
				refreshColumn(i);
			}


		}
		else
		{
			throw "cached in invalid column";
		}

	}

	// Compute the kernel function
	double compute(size_t i, size_t j) const
	{
		return m_kernel.compute(m_P, m_T, i, j);
	}

	// Kernel function
	const Kernel& m_kernel;

	// Contains mapping of external indices to caching structure and active indices
	vector<list<CacheLine>::iterator> m_colIndices;
	vector<size_t> m_rowIndices;

	// Contains list of active indices
	vector<size_t> m_activeIndices;
	size_t m_activeLength;

	// List structure implements LRU caching mechanism
	list<CacheLine> m_cache;
	list<CacheLine>::iterator NULLITEM;

	// Referenced data used for computing Hessian
	const Array<double, 2>& m_P;
	const Array<double, 1>& m_T;

	// Pre-caching of diagonal elements
	vector<double> m_cacheDiagonal;

	// Maximum caching size
	size_t m_cacheSize;

	// Currently allocated cache size
	size_t m_currentAllocSize;

};






