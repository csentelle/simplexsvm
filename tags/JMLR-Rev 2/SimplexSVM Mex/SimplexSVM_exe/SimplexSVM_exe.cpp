// SimplexSVM_exe.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "MLDataSet.h"
#include "MLDataFile.h"
#include "GenArrayList.h"
#include "StandaloneArrayList.h"
#include "ProxyStream.h"

// Define the external entry function that will be called by this function.
extern void EntryFunction(ProxyStream& os, 
                          const GenArrayList& InputList, 
                          GenArrayList& OutputList);

extern void SimplexSVM(ProxyStream& os, 
                const Array<double, 2>& P,
                const Array<double, 1>& T,
                double c,
                double gamma,
                int kernelType,
                int verb,
                int working_size,
                int shrinking_iter,
                Array<double, 1>& alpha,
                double& b,
                double& t,
                int& iter);


int _tmain(int argc, _TCHAR* argv[])
{

    try
    {
        ProxyStream pos(cout);

        Array<double, 2> P;
        Array<double, 2> T;

        MLDataFile<double, UINT32> file("spam");
        MLData<double, UINT32> data;

        // Read in the data
        file.readDatabase(data);

        // Obtain the data
        P.reference(data.getAllNumericalAttributes());
        T.reference(data.getAllNumericalTargets());


        TinyVector<int, 2> ex = P.extent();

        P.transposeSelf(secondDim, firstDim);
        T.transposeSelf(secondDim, firstDim);
        Array<double, 1> T2 = T(0, Range::all());

        //StandaloneArrayList inputs(6);
        //StandaloneArrayList outputs(4);
        //
        //(GenArray&)(inputs[0]) = P;
        //(GenArray&)(inputs[1]) = T2;
        //(GenArray&)(inputs[2]) = 100.0;
        //(GenArray&)(inputs[3]) = 0.01;
        //(GenArray&)(inputs[4]) = 1;
        //(GenArray&)(inputs[5]) = 100;

        Array<double, 1> alpha;
        double b;
        double t;
        int iter;

        SimplexSVM(pos, P, T2, 1.0, 0.00333, 1, 0, 100, 200, alpha, b, t, iter);

        pos << infolevel(0) << "b = " << b << endl;
        pos << infolevel(0) << "t = " << t << endl;
        pos << infolevel(0) << "iter = " << iter << endl;

//        EntryFunction(pos, inputs, outputs);
        char ch;
         cin >> ch;


	}
    catch(FileException& e)
    { 
        cout << e.what();
    }
    catch(InputParamException& e)
    {
        cout << e.what();
    }
    catch(exception& e)
	{
		cout << e.what();
	}
	catch(...)
	{
		cout << ("Unknown Error");
	}
	return 0;
}

