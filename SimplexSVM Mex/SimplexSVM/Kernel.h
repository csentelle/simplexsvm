#pragma once

// For Blitz Array class.
#include <blitz/array.h>         
BZ_USING_NAMESPACE(blitz) 
#include <list>
#include <vector>
using namespace std;

//#define PROFILE	&Profile
#undef PROFILE
#include "hwprof.h"

extern CHWProfile Profile; 

template<typename T>
inline void INSERT_ELEMENTS(T& coll, int first, int last)
{
	for (int i = first; i <= last; ++i)
		coll.insert(coll.end(),i);
};

template<typename T>
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
template<typename KType = float>
struct CacheLine
{
	// Default constructor
	CacheLine() 
		: m_buffer(NULL), m_len(0), m_index(0), m_validlen(0), m_lock(false) {};

	CacheLine(KType* buffer, size_t len, size_t index) 
		: m_buffer(buffer), m_len(len), m_index(index), m_validlen(0), m_lock(false) {};

	KType* m_buffer;
	size_t m_len;	
	size_t m_index;
	size_t m_validlen; 
	bool m_lock;
};


template<typename DataType = double, typename KType = float>
class Kernel
{
public: 
	
	Kernel(const Array<DataType, 2>& P, const Array<DataType, 1>& T) : m_P(P), m_T(T) {}
	virtual ~Kernel() {}

	virtual void compute(const size_t i, 
						 const vector<size_t>& indices, 
						 const size_t len, 
						 KType* buffer) const = 0; 

	virtual KType operator()(const size_t i, 
							const size_t j) const = 0;

	int getDim(void) const { return m_P.extent(0) ; } 
	int getNum(void) const { return m_T.size(); } 

protected:

	const Array<DataType, 2>& m_P;
	const Array<DataType, 1>& m_T;
};

template<typename DataType = double, typename KType = float>
class LinearKernel : public Kernel<DataType, KType>
{
public:

	LinearKernel(const Array<DataType, 2>& P, const Array<DataType, 1>& T) : Kernel(P, T) {};
	virtual ~LinearKernel(void) {};

	virtual KType operator()(const size_t i, 
							  const size_t j) const
	{
		size_t N = m_P.extent(0);

		DataType sum = 0.0;

		const DataType *points1 = &m_P.data()[ i * N ];
		const DataType *points2 = &m_P.data()[ j * N ];

		for (size_t k = 0; k < N; k++)
			sum += points1[k] * points2[k];

		return static_cast<KType>(sum * m_T((int)i) * m_T((int)j));

	}

	void compute(const size_t i, const vector<size_t>& indices, 
				 const size_t len, KType* buffer) const
	{
		size_t N = m_P.extent(0);

		for (int l = 0; l < len; ++l)
		{
			DataType sum = 0.0;

			const DataType *points1 = &m_P.data()[ i * N ];
			const DataType *points2 = &m_P.data()[ indices[l] * N ];

			for (size_t k = 0; k < N; k++)
				sum += points1[k] * points2[k];

			buffer[l] = static_cast<KType>(sum * m_T((int)i) * m_T((int)indices[l]));
		}
	}
};


template<typename DataType = double, typename KType = float>
class RBFKernel: public Kernel<DataType, KType>
{
public:

	RBFKernel(const Array<DataType, 2>& P, const Array<DataType, 1>& T, DataType fpGamma)
		: Kernel(P, T), m_fpGamma(fpGamma) {};
	virtual ~RBFKernel(void) {};

	virtual KType operator()(const size_t i, 
							  const size_t j) const
	{
		size_t N = m_P.extent(0);

		DataType sum = 0.0;
		DataType diff = 0.0;

		const DataType *points1 = &m_P.data()[ i * N ];
		const DataType *points2 = &m_P.data()[ j * N ];

		for (size_t k = 0; k < N; k++)
		{
			diff = points1[k] - points2[k];
			sum += diff * diff;
		}

		return static_cast<KType>(exp(-m_fpGamma * sum) * m_T((int)i) * m_T((int)j));

	}

	void compute(const size_t i, 
				 const vector<size_t>& indices, 
				 const size_t len, 
				 KType* buffer) const
	{
		size_t N = m_P.extent(0);	

		const DataType* points1 = &m_P.data()[ i * N ];
		const DataType* points2buf = m_P.data();

		DataType sum = 0.0;
		DataType diff = 0.0;
		
		for (int l = 0; l < len; ++l)
		{

			sum = 0.0;
			diff = 0.0;

			const DataType* points2 = &points2buf[ indices[l] * N ];

			for (size_t k = 0; k < N; k++)
			{
				diff = points1[k] - points2[k];
				sum += diff * diff;
			}

			buffer[l] = 
				static_cast<KType>(exp(-m_fpGamma * sum) * m_T((int)i) * m_T((int)indices[l]));
		}
	}

private:

	const DataType m_fpGamma;

};


template<typename DataType = double, typename KType = float>
class KernelCache
{
public:

	typedef typename list<CacheLine<KType> >::iterator list_iter;

	KernelCache(const Kernel<DataType, KType>& kernel,				
				unsigned int cacheSize = 100) 
		: m_kernel(kernel)
		, m_cacheSize(cacheSize * 1024 * 1024)
		, m_currentAllocSize(0)
		, m_colIndices(kernel.getNum())
		, m_activeLength(kernel.getNum())
		, m_cacheDiagonal(kernel.getNum())
	{

		for (int i = 0; i < kernel.getNum(); i++)
		{
			m_cacheDiagonal[i] = m_kernel(i, i);
		}

		// Initialize the active list and all of the row indices
		INSERT_ELEMENTS(m_activeIndices, 0, kernel.getNum() - 1);
		INSERT_ELEMENTS(m_rowIndices, 0, kernel.getNum() - 1);

		// Add a NULL reference. The item at the end of the list will contain a pointer to
		// an empty buffer. This is used as a NULL reference since iterators cannot be
		// specified as NULL.
		m_cache.push_back(CacheLine<KType>());
		NULLITEM = m_cache.begin();

		for (size_t i = 0; i < m_colIndices.size(); i++)
			m_colIndices[i] = NULLITEM;

	}


	virtual ~KernelCache(void) 
	{
		// Clear memory allocated by cache. Remaining objects should be cleaned up
		// by default destructors
		clearCache();
	}

	KType operator()(size_t i, size_t j) 
	{
		updateColumn(i);
		return getCachedItem(i,j);
	}

	KType getDirect(size_t i, size_t j)
	{		
		return m_kernel(i, j);
	}

	const size_t& indexMap(size_t i) const
	{
		return m_rowIndices[i];
	}

	size_t getActiveSize() const
	{
		return m_activeLength;
	}

	KType getQss(size_t i)
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
		//BEGIN_PROF("REMOVE ACTIVE INDEX");
		// Swap this item out with the last element in the list.
		size_t idx1 = m_rowIndices[i];
		size_t idx2 = m_activeLength-1;

		// Proceed through the items in each cache line and swap the computed value to be 
		// removed with the last in the active list
		for (list_iter pos = m_cache.begin(); pos != NULLITEM; )
		{
			// If the buffer length is not the current active length, go ahead and 
			// remove the item from the cache as an alternative to reallocating and 
			// refreshing. We assume that the item hasn't been accessed for a while.
			// Most likely the item has not been accessed for a while.
			if ((*pos).m_len < m_activeLength)
			{
				//BEGIN_PROF("REMOVE CACHE ENTRY");
				pos = removeCacheEntry(pos);
				//END_PROF();

			}
			else
			{
				//BEGIN_PROF("SWAP DATA IN BUFFERS");
				swapdata((*pos).m_buffer[idx1], (*pos).m_buffer[idx2]);
				++pos;
				////END_PROF();
			}
		}

		m_rowIndices[m_activeIndices[idx1]] = idx2;
		m_rowIndices[m_activeIndices[idx2]] = idx1;

		// Swap entries in the active indices to place the removed variable at
		// the end of the current active list.
		swapdata(m_activeIndices[idx1], m_activeIndices[idx2]);

		// Reduce the length of the active indices
		m_activeLength--;
		//END_PROF();
	}

	bool isCached(size_t i)
	{
		return m_colIndices[i]->m_buffer != NULL;
	}

	inline KType getUnsafeCachedItem(size_t i, size_t j)
	{
		// Assumes the column is already cached!
		return getCachedItem(i, j);

	}

	// The user is allowed to force a cache of a column, where subsequently, 
	// a call to getUnsafeCachedItems can be called. The getUnsafeCachedItems
	// is used for fast access where it is assumed this is called first!
	void updateColumn(size_t i)
	{
		//BEGIN_PROF("updateColumn");
		// If the column is already cached, just update the column, otherwise
		// prepare to cache the column.
		if (NULL != m_colIndices[i]->m_buffer)
		{
			refreshColumn(m_colIndices[i]);
		}
		else
		{
			cacheColumn(i);
		}
		//END_PROF();
	}

	const vector<size_t>& getRowIndices() const { return m_rowIndices; }
	
	//
	// This method allows direct access to a precomputed kernel buffer and 
	// locks the buffer to prevent future caching from replacing the buffer. 
	// Locking multiple buffers can prevent future caching causing an exception
	// to be thrown.
	//
	const KType* getColumnAndLock(size_t i) 
	{ 
		updateColumn(i);
		m_colIndices[i]->m_lock = true;
		return m_colIndices[i]->m_buffer; 
	}

	void unlockColumn(size_t i)
	{
		m_colIndices[i]->m_lock = false;
	}

	void unlockAllColumns(void) 
	{
		for (list_iter pos = m_cache.begin(); pos != NULLITEM; ++pos )
			pos->m_lock = false;
	}

protected:


	inline KType getCachedItem(size_t i, size_t j)
	{
		assert(isCached(i) && m_rowIndices[j] < m_activeLength);
		return m_colIndices[i]->m_buffer[m_rowIndices[j]];
	}

	// Proceed through the cache and clean up all cache lines
	void clearCache()
	{
		for (list_iter pos = m_cache.begin(); pos != NULLITEM; )
			pos = removeCacheEntry(pos);

	}


	void refreshColumn(list_iter& posCacheLine)
	{
		//BEGIN_PROF("REFRESH COLUMN");
		assert(posCacheLine != NULLITEM);

		// Move the item to the front of the list
		CacheLine<KType> cacheline = *posCacheLine;
		
		m_cache.erase(posCacheLine);
		m_cache.push_front(cacheline);
		
		posCacheLine = m_cache.begin();

		// Determine if the size is correct.
		if (posCacheLine->m_validlen < m_activeLength)
		{
			if (posCacheLine->m_len < m_activeLength)
				reallocColumn(*posCacheLine);

			// Really only need to compute a partial column for speed
			computeColumn(posCacheLine);

		}
		//END_PROF();
	}

	void reallocColumn(CacheLine<KType>& cacheline)
	{
		//BEGIN_PROF("REALLOC COLUMN");
		// Note, we are assuming that the item being refreshed is not at 
		// risk of being deleted, below.

		size_t diffLen = (m_activeLength - cacheline.m_len) * sizeof(KType); 
		if ( diffLen + m_currentAllocSize > m_cacheSize )
		{
			// Need to free some additional memory, by the mount of diffLen.
			list_iter pos = NULLITEM;
			pos--;
							
			while (diffLen + m_currentAllocSize > m_cacheSize && pos != m_cache.begin())
			{
				m_currentAllocSize -= pos->m_len * sizeof(KType);
				pos = removeCacheEntry(pos);
				--pos;
			}
			
		}

		m_currentAllocSize -= cacheline.m_len * sizeof(KType);

		// We need to resize the column
		cacheline.m_buffer = 
			(KType*)realloc(cacheline.m_buffer, m_activeLength * sizeof(KType));
		cacheline.m_len = m_activeLength;

		m_currentAllocSize += m_activeLength * sizeof(KType);				

		assert(m_currentAllocSize <= m_cacheSize);
		//END_PROF();
	}

	list_iter removeCacheEntry(list_iter& pos)
	{

		if (pos->m_lock) return(pos);

		// Remove reference from the col indices
		m_colIndices[pos->m_index] = NULLITEM;

		// Free the associated buffer
		free(pos->m_buffer);

		// erase item from cache and return iterator to next position
		return m_cache.erase(pos);
	}

	void computeColumn(list_iter& posCacheLine)
	{		
		//BEGIN_PROF("COMPUTE COLUMN");
		KType* column = posCacheLine->m_buffer;
		size_t idx = posCacheLine->m_index;

		m_kernel.compute(idx, m_activeIndices, m_activeLength, column);

		posCacheLine->m_validlen = m_activeLength;
		//END_PROF();
	}


	// Cache a column of the kernel matrix. Only the portion corresponding
	// to the active indices is cached.
	void cacheColumn(size_t i) 
	{
		BEGIN_PROF("CACHE COLUMN");
		if (NULLITEM == m_colIndices[i])
		{
			size_t len = m_activeLength * sizeof(KType);

			// If there is sufficient cache left over
			if (len + m_currentAllocSize <= m_cacheSize)
			{
				// Just allocate a buffer.
				m_cache.push_front(CacheLine<KType>((KType*)malloc(len), 
											 m_activeLength, i));

				m_colIndices[i] = m_cache.begin();
				m_currentAllocSize += len;
				computeColumn(m_colIndices[i]);

			}
			else
			{
				//
				// First, determine how much memory is needed and remove items
				// until required memory is obtained.
				//
				// Delete the LRU and realloc for the purposes of this item. Note
				// that the very last item is the NULLITEM. 
				//
				list_iter pos = NULLITEM;
				
				if (pos != m_cache.begin())
					--pos;

				while (len + m_currentAllocSize > m_cacheSize && pos != m_cache.begin())
				{
					if (!pos->m_lock)
					{
						// reduce current allocation size
						m_currentAllocSize -= pos->m_len * sizeof(KType);

						// Remove entry
						pos = removeCacheEntry(pos);
					}

					// point to previous item instead of next
					--pos;
				}
				
				if (pos == NULLITEM || len + m_currentAllocSize > m_cacheSize)
					throw exception("Unable to allocate cache line, increase memory");

				m_colIndices[pos->m_index] = NULLITEM;
				pos->m_index = i;	
				pos->m_validlen = 0;
				m_colIndices[i] = pos;
			
				// This will force a reallocation and recompute on the column
				refreshColumn(m_colIndices[i]);
				
			}
		}
		else
		{
			throw exception("cached in invalid column");
		}
		END_PROF();
	}

	void clearMemory(size_t len)
	{

	}

	// Kernel function
	const Kernel<DataType, KType>& m_kernel;

	// Contains mapping of external indices to caching structure and active indices
	vector<list_iter> m_colIndices;
	vector<size_t> m_rowIndices;

	// Contains list of active indices
	vector<size_t> m_activeIndices;
	size_t m_activeLength;

	// List structure implements LRU caching mechanism
	list<CacheLine<KType> > m_cache;
	list_iter NULLITEM;

	// Pre-caching of diagonal elements
	vector<KType> m_cacheDiagonal;

	// Maximum caching size
	size_t m_cacheSize;

	// Currently allocated cache size
	size_t m_currentAllocSize;

};






