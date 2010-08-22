

#include "stdlib.h"

#define PROFILE	&Profile
#include "HWProf.h"


CHWProfile Profile;


void main()
{
	BEGIN_PROF("Test");
	END_PROF();

	Profile.dumpprint();
	Profile.reset();
};



