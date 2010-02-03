#pragma once

// For Blitz Array class.
#include <blitz/array.h>         
BZ_USING_NAMESPACE(blitz) 

struct SparseNode
{
    SparseNode(double value, int index) 
        : m_value(value), m_index(index) {}

    double m_value;
    int m_index;
};

#include <vector>
using namespace std;
class Kernel
{
public:

	Kernel(const Array<double, 2>& P, 
           const Array<double, 1>& T,
           int cacheSize = 4000) 
    : m_P(P)
    , m_T(T)
    {


        //m_cache.resize(T.size());
        //m_cachess.resize(T.size());
        m_cachess = new double[T.size()];
        m_cache = new double*[T.size()];

        for (int i = 0; i < T.size(); i++)
            m_cache[i] = NULL;
            
        m_cacheLineSize = cacheSize; //100e6 / (T.size() * 8);

        m_cacheLines = new double[m_T.size() * m_cacheLineSize];
        m_cacheLineIdx = 0;
        m_cacheOwners.resize(m_cacheLineSize);

        m_nCacheCount = 0;

    };


    virtual ~Kernel(void) 
    {

        delete [] m_cacheLines;
        delete [] m_cachess;
        delete [] m_cache;

    };

	virtual double operator()(int i, int j) = 0;
    virtual void cacheColumn(int i) = 0;

    virtual double compute(int i, int j) const = 0;
    
    double** get_cache()
    {
        return m_cache;
    }

    inline double* operator[](int i)
    {
        if (!m_cache[i])
        {
            cacheColumn(i);
        }

        return m_cache[i];
    }

    inline double getQss(int i)
    {
        return m_cachess[i];
    }

    int m_nCacheCount;

protected:

    double* obtainCacheLine(int i)
    {
        if (m_cacheLineIdx >= m_cacheLineSize)
        {
            // Generate a random number and replace someone
            int repIdx = ++m_cacheLineIdx % m_cacheLineSize;
            
            m_cache[m_cacheOwners[repIdx]] = NULL;
            m_cacheOwners[repIdx] = i;
            
            return &m_cacheLines[repIdx * m_T.size()];
           
        }
        else
        {
            m_cacheOwners[m_cacheLineIdx] = i;
            return &m_cacheLines[m_cacheLineIdx++ * m_T.size()];
   
        }   
    }

    void storeSparseRepresentation(const Array<double, 2>& P)
    {
        m_sparseP.resize(P.extent(1));
        
        for (int i = 0; i < m_sparseP.size(); i++)
        {
            for (int k = 0; k < P.extent(0); k++)
            {
				const double val = P(k, i);
                if (abs(val) > 1e-6 )
                {
                    m_sparseP[i].push_back(SparseNode(val, k));
                }
            }
            m_sparseP[i].push_back(SparseNode(0.0, -1));
        }
    }

    // Stores the data in sparse format.
    vector<vector<SparseNode> > m_sparseP;
    vector<int> m_cacheOwners;

    const Array<double, 2>& m_P;
    const Array<double, 1>& m_T;

    double** m_cache;
    double* m_cachess;
    double* m_cacheLines;
    int m_cacheLineIdx;
    int m_cacheLineSize;
};

class RBFKernelSparse : public Kernel
{
public:

    RBFKernelSparse(const Array<double, 2>& P, 
              const Array<double, 1>& T, 
              double fpGamma,
              int cacheSize) 
		: Kernel(P, T, cacheSize)
        , m_fpGamma(fpGamma) 
    {
        for (int i = 0; i < T.size(); i++)
            m_cachess[i] = compute(i, i); 

        Kernel::storeSparseRepresentation(P);

    };

	virtual ~RBFKernelSparse(void) {};


    virtual void cacheColumn(int i) 
    {
        Kernel::m_nCacheCount++;

        if (NULL == m_cache[i])
        {
            double sum = 0.0;
            int N = m_P.extent(0);

            m_cache[i] = obtainCacheLine(i);

            for (int j = 0; j < m_T.size(); j++)
            {                
                sum = 0.0;

                int idxi = 0;
                int idxj = 0;

                while (m_sparseP[i][idxi].m_index != -1 && m_sparseP[j][idxj].m_index != -1)
                {
                    if (m_sparseP[i][idxi].m_index == m_sparseP[j][idxj].m_index)
                    { 
                        double diff =
                            m_sparseP[i][idxi].m_value - m_sparseP[j][idxj].m_value;
                        
                        sum += diff * diff;
                        
                        idxi++;
                        idxj++;
                    }
                    else if (m_sparseP[i][idxi].m_index > m_sparseP[j][idxj].m_index)
                    {
                        sum += 
                            m_sparseP[j][idxj].m_value * m_sparseP[j][idxj].m_value;
                        
                        idxj++;
                    }
                    else if (m_sparseP[j][idxj].m_index > m_sparseP[i][idxi].m_index)
                    {
                        sum += m_sparseP[i][idxi].m_value * m_sparseP[i][idxi].m_value;
                        idxi++;
                    }
                }

                while (m_sparseP[j][idxj].m_index != -1)
                {
                        sum += m_sparseP[j][idxj].m_value * m_sparseP[j][idxj].m_value;
                        idxj++;
                }

                while (m_sparseP[i][idxi].m_index != -1)
                {
                        sum += m_sparseP[i][idxi].m_value * m_sparseP[i][idxi].m_value;
                        idxi++;
                }

                for (int k = 0; k < N; k++)
                {
                    double diff = 0.0;
					diff = m_P(k, i) - m_P(k, j);
                    sum += diff * diff;
                }
                
                m_cache[i][j] = exp(-m_fpGamma * sum) * m_T(i) * m_T(j);

            }
        }
    }

    double compute(int i, int j) const
    {
        double sum = 0.0;
        for (int k = 0; k < m_P.extent(0); k++)
        {
            double diff = m_P(k, i) - m_P(k, j);
            sum += diff * diff;
        }

		return exp(-m_fpGamma * sum) * m_T(i) * m_T(j);
    }

	virtual double operator()(int i, int j) 
	{
        if (m_cache[i])
        {
            return m_cache[i][j];
        }
        else if (m_cache[j])
        {
            return m_cache[j][i];
        }
        else
        {
            cacheColumn(i);
            return m_cache[i][j];
        }
	}
	
private:

	const double m_fpGamma;


};

class RBFKernel : public Kernel
{
public:

    RBFKernel(const Array<double, 2>& P, 
              const Array<double, 1>& T, 
              double fpGamma,
              int cacheSize) 
		: Kernel(P, T, cacheSize)
        , m_fpGamma(fpGamma) 
    {
        for (int i = 0; i < T.size(); i++)
            m_cachess[i] = compute(i, i); 
    };

	virtual ~RBFKernel(void) {};

    virtual void cacheColumn(int i) 
    {
        Kernel::m_nCacheCount++;

        if (NULL == m_cache[i])
        {

			double sum = 0.0;
            int N = m_P.extent(0);

			const double *points1 = &m_P.data()[ i * N ];
			const double *points2buf = m_P.data();
			
            m_cache[i] = obtainCacheLine(i);

            for (int j = 0; j < m_T.size(); j++)
            {                
                sum = 0.0;
				double diff = 0.0;
				const double* points2 = &points2buf[j * N];
				//Array<double, 1>& points1 = m_P(Range::all(), i);
				//Array<double, 1>& points2 = m_P(Range::all(), j);

                for (int k = 0; k < N; k++)
                {
                    //diff = m_P(k, i) - m_P(k, j);
					//diff = points1(k) - points2(k);
                    diff = points1[k] - points2[k];
					sum += diff * diff;
                }
                
                m_cache[i][j] = exp(-m_fpGamma * sum) * m_T(i) * m_T(j);

            }
        }
    }

    double compute(int i, int j) const
    {
        double sum = 0.0;
        for (int k = 0; k < m_P.extent(0); k++)
        {
            double diff = m_P(k, i) - m_P(k, j);
            sum += diff * diff;
        }

		return exp(-m_fpGamma * sum) * m_T(i) * m_T(j);
    }

	virtual double operator()(int i, int j) 
	{
        if (m_cache[i])
        {
            return m_cache[i][j];
        }
        else if (m_cache[j])
        {
            return m_cache[j][i];
        }
        else
        {
            cacheColumn(i);
            return m_cache[i][j];
        }
	}
	
private:

	const double m_fpGamma;


};

class LinearKernelSparse : public Kernel
{
public:
	LinearKernelSparse(const Array<double,2>& P, const Array<double, 1>& T, int cacheSize)
        : Kernel(P, T, cacheSize)
    {
        for (int i = 0; i < T.size(); i++)
            m_cachess[i] = compute(i, i);

        Kernel::storeSparseRepresentation(P);
    };

	virtual ~LinearKernelSparse() 
    {

    };

    virtual void cacheColumn(int i) 
    {
        if (NULL == m_cache[i])
        {
            double sum = 0.0;
            int N = m_P.extent(0);

            m_cache[i] = obtainCacheLine(i);

            for (int j = 0; j < m_T.size(); j++)
            {                
                sum = 0.0;

                int idxj = 0;
                int idxi = 0;

                while (m_sparseP[i][idxi].m_index != -1 && m_sparseP[j][idxj].m_index != -1)
                {
                    if (m_sparseP[i][idxi].m_index == m_sparseP[j][idxj].m_index)
                    {
                        sum += 
                            m_sparseP[i][idxi].m_value * m_sparseP[j][idxj].m_value;
                        
                        idxj++;
                        idxi++;
                    }
                    else if (m_sparseP[i][idxi].m_index > m_sparseP[j][idxj].m_index)
                    {
                        idxj++;
                    }
                    else if (m_sparseP[j][idxj].m_index > m_sparseP[i][idxi].m_index)
                    {
                        idxi++;
                    }
                }

                m_cache[i][j] = sum * m_T(i) * m_T(j);

            }
        }
    };

    inline double compute(int i, int j) const
    {
        double sum = 0.0;
        for (int k = 0; k < m_P.extent(0); k++)
            sum += m_P(k, i) * m_P(k, j);

        return sum * m_T(i) * m_T(j);
    }

	virtual double operator()(int i, int j) 
	{
        if (m_cache[i])
        {
            return m_cache[i][j];
        }
        else if (m_cache[j])
        {
            return m_cache[j][i];
        }
        else
        {
            cacheColumn(i);
            return m_cache[i][j];
        }
	}

private:


};


class LinearKernel : public Kernel
{
public:
	LinearKernel(const Array<double,2>& P, const Array<double, 1>& T, int cacheSize)
        : Kernel(P, T, cacheSize)
    {
        for (int i = 0; i < T.size(); i++)
            m_cachess[i] = compute(i, i);

    };

	virtual ~LinearKernel() 
    {

    };



    virtual void cacheColumn(int i) 
    {
        if (NULL == m_cache[i])
        {
            double sum = 0.0;
            int N = m_P.extent(0);

            m_cache[i] = obtainCacheLine(i);

            for (int j = 0; j < m_T.size(); j++)
            {                
                sum = 0.0;
				
				const double *points1 = &m_P.data()[ i * N ];
				const double *points2 = &m_P.data()[ j * N ];

				for (int k = 0; k < N; k++)
					sum += points1[k] * points2[k];
                    //sum += m_P(k, i) * m_P(k, j);
                
                m_cache[i][j] = sum * m_T(i) * m_T(j);

            }
        }
    };

    inline double compute(int i, int j) const
    {
        double sum = 0.0;
        for (int k = 0; k < m_P.extent(0); k++)
            sum += m_P(k, i) * m_P(k, j);

        return sum * m_T(i) * m_T(j);
    }

	virtual double operator()(int i, int j) 
	{
        if (m_cache[i])
        {
            return m_cache[i][j];
        }
        else if (m_cache[j])
        {
            return m_cache[j][i];
        }
        else
        {
            cacheColumn(i);
            return m_cache[i][j];
        }
	}

private:


};