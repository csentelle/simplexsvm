#include "MLDataSet.h"
#include "GenArrayList.h"
#include "ProxyStream.h"
#include "Kernel.h"
#include "SVM.h"
#include "time.h"

#include <string>
using namespace std;

//Strong typed entry function (one input, one output)
void SimplexSVM(ProxyStream& os, 
                const Array<double, 2>& P,
                const Array<double, 1>& T,
                double c,
                double gamma,
                int kernelType,
				double tol, 
                int verb,
                int working_size,
                int shrinking_iter,
                Array<double, 1>& alpha,
                double& b,
                double& t,
                int& iter)
{

    os.setVerbosity(verb);

    alpha.resize(T.shape());
    alpha(Range::all()) = 0.0;

    KernelCache<double, float>* kernelCache = NULL;
	Kernel<double, float>* kernel = NULL;

    try
    {
        if (kernelType == 0)
        {
			kernel = new LinearKernel<double, float>(P, T);
            kernelCache = new KernelCache<double, float>(*kernel, 500);
        } 
        else if (kernelType == 1)
        {	
			kernel = new RBFKernel<double, float>(P, T, gamma);
            kernelCache = new KernelCache<double, float>(*kernel, 500);
        }
        else if (kernelType == 2)
        {
            throw new string("Sparse kernel unimplemented");
        }
        else if (kernelType == 3)
        {
            throw new string("Sparse kernel unimplemented");        
        }
    
	    SVM svm_classifier(*kernelCache, c, tol, os);

        svm_classifier.train(P, 
                             T, 
                             alpha, 
                             b, 
                             iter, 
                             t, 
                             working_size, 
                             shrinking_iter);

        delete kernelCache;
		delete kernel;

    }
    catch(...)
    {
        delete kernel;
        throw;
    }
};

//entry function with variable parameter list
void EntryFunction(ProxyStream& os, 
                   const GenArrayList& inputList, 
                   GenArrayList& outputList)
{
	if(inputList.getNumberOfArrays() < 8 || inputList.getNumberOfArrays() > 9)
		throw exception("Invalid number of arguments");

    int verb = 0;
    if (inputList.getNumberOfArrays() == 9)
    {
        verb = (int)inputList[8].convertTo<double, 1>(9, "verb")(0);
    }  

    Array<double, 1> alpha;
    double t = 0.0;
    int iter = 0;
    double b = 0.0;

	SimplexSVM(os, 
            inputList[0].convertTo<double, 2>(1, "P"),
			inputList[1].convertTo<double, 1>(2, "T"),
            inputList[2].convertTo<double, 1>(3, "C")(0),
            inputList[3].convertTo<double, 1>(4, "g")(0),
            inputList[4].convertTo<int, 1>(5, "ktype")(0),
			inputList[5].convertTo<double, 1>(6, "tol")(0),
            verb, 
            inputList[6].convertTo<int, 1>(7, "working_size")(0),
            inputList[7].convertTo<int, 1>(8, "shrink_iter")(0),
            alpha,
            b,
            t,
            iter);

    if (outputList.getNumberOfArrays() >= 4)
    {
        outputList[0] = alpha;
        outputList[1] = b;
        outputList[2] = t;
        outputList[3] = iter;
    }
    else
    {
        throw InputParamException("Insufficient output arguments");
    }

};