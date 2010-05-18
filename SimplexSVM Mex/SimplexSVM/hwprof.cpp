

/* *************************************************************************

Source Profiler -- Copyright (C) 1995, 1996, 1997, 1998 - Harald Hoyer
-----------------------------------------------------

************************************************************************* */

//#include "hwlib.h"
#include "hwprof.h"
//#include "str.h"
#pragma warning (disable : 4201) // C4201 nonstandard extension used : nameless struct/union
#include "windows.h"
#pragma warning (default : 4201) // C4201 nonstandard extension used : nameless struct/union
#include "proxystream.h"

__int64 Ticks()
{
	LARGE_INTEGER x;
	BOOL rc = ::QueryPerformanceCounter(&x);
	return x.QuadPart;
};


__int64 TicksPerSecond()
{
	LARGE_INTEGER x;
	BOOL rc = ::QueryPerformanceFrequency(&x);
	return x.QuadPart;
};


class CHWProfileEntry
{
public:
	CHWProfileEntry* Next;
	const char *const FileName;
	const int Line;
	const char *const Flag;

	__int64 LastTick;
	__int64 xTicks;
	__int64 Calls;

	CHWProfileEntry(CHWProfileEntry*x,const char*fn,int l,const char*fl)
	: Next(x), FileName(fn), Line(l), Flag(fl), xTicks(0), Calls(0)
	{};

	void Suspend() {xTicks += ::Ticks() - LastTick;};
	void Resume () {LastTick = ::Ticks();};

	void Suspend(__int64 Correction) {xTicks += ::Ticks() - LastTick - Correction;};
	void Resume (__int64 Correction) {LastTick = ::Ticks() + Correction;};
};


class CHWProfileStack
{
public:
	CHWProfileStack* Next;
	CHWProfileEntry &Top;
	CHWProfileStack(CHWProfileStack*n, CHWProfileEntry&t): Next(n), Top(t) {};
};


CHWProfile::CHWProfile()
: First(NULL)
,Current(NULL) 
{
	for(int ixxx=0; ixxx<1000; ixxx++)
	{
		Start("***Calibrate***",-1,"suspend time");
		Start("***Calibrate***",-1,"resume time");
		End();
		End();
	};
	Start("***Calibrate***",-1,"calibration");
	End();
};


void CHWProfile::Start(const char*FileName,int Line, const char*Flag)
{       
	static __int64 SuspendTime = 0;
	static __int64 ResumeTime = 0;
	
	if(Current)
		Current->Top.Suspend(SuspendTime);
		
	CHWProfileEntry* NewEntry = First;

	for(NewEntry = First;
		NewEntry && (strcmp(FileName,NewEntry->FileName) || Flag != NewEntry->Flag);
		NewEntry = NewEntry->Next
	);

	if(NewEntry==0)
	{
		First = new CHWProfileEntry(First, FileName,Line, Flag);
		NewEntry = First;
	};

	Current = new CHWProfileStack(Current,*NewEntry);
    
    // Obtain overhead of sample
	if(Line == -1 && Flag && *Flag == 'c')
	{
		for(CHWProfileEntry* Entry = First;Entry;Entry = Entry->Next)
		{
			if( Entry->Line == -1 )
			{
				switch(*Entry->Flag)
				{
					case 's':
						SuspendTime = Entry->xTicks / Entry->Calls;  
						break;
					case 'r':
						ResumeTime = Entry->xTicks / Entry->Calls;
						break;
				};
			};
		};  
		SuspendTime -= ResumeTime;
	};	
	
	NewEntry->Calls++;
	NewEntry->Resume(ResumeTime);
};


void CHWProfile::End()
{
	if(Current)
	{
		Current->Top.Suspend();
		CHWProfileStack* ccc = Current;
		Current = ccc->Next;
		delete ccc;
		if(Current) Current->Top.Resume();
	};
};


void CHWProfile::reset()
{
	while (Current) End();

	while(First)
	{
		CHWProfileEntry*eee = First;
		First = First->Next;
		delete eee;
	};
};


void dumpprint(ProxyStream& os, const char*Text)
{
	os << infolevel(0) << Text;
	//::OutputDebugString(Text);
};

            
int dumpprint(ProxyStream& os, __int64 x)
{
	char Text[100];
	_ltoa_s(int(x),Text,10);
	dumpprint(os, Text);
	return (int)::strlen(Text);
};

            
void dumpprint(ProxyStream& os, __int64 x, int Scale, const char*xText)
{
	char Text[100];

	_ltoa_s(int(x),Text + 1,99,10);

	int Dot = 1;
	for(Dot=1; Text[Dot]; Dot++);
	Dot -= Scale;
	Dot--;
	for(int i=0; i < Dot; i++)
		Text[i] = Text[i+1];
	if(Scale > 0) Text[Dot] = '.';	
	else Text[Dot] = 0;
	dumpprint(os,Text);
	dumpprint(os,xText);
};

            
void dumpprinttime(ProxyStream& os, __int64 x, __int64 q)
{       
	double xx = double(x)/(double(TicksPerSecond())*double(q)*3600.0);
	if(xx > 10.0)
	{
		dumpprint(os,int(xx),0,"h");
		return;
	};

	if(xx >= 1.0)
	{
		int h = int(xx);
		dumpprint(os,h);
		dumpprint(os,":");
		xx-=h;
		xx*=60;
		int m = int(xx);
		if(m<10) dumpprint(os,"0");
		dumpprint(os,m);
		return;
	};

	xx*=60;
	if(xx >= 1.0)
	{
		dumpprint(os,":");
		int m = int(xx);
		if(m<10) dumpprint(os,"0");
		dumpprint(os,m);
		dumpprint(os,":");
		xx-=m;
		xx*=60;
		int s = int(xx);
		if(s<10) dumpprint(os,"0");
		dumpprint(os,s);
		return;
	};

	xx*=600;

	char* Unit[6] = {"s","ms","µs","ns","ps","fs"};
	for(int i=1;i<18;i++)
	{
		if(xx >= 100.0)
		{
			dumpprint(os,int(xx),i%3,Unit[i/3]);
			return;
		};
		xx*=10;
	};
	dumpprint(os,"0");
};

            
void CHWProfile::dumpprint(ProxyStream& os, int Hide)const
{
	if(First == NULL)
	{
		::dumpprint(os,"\n\r=========== Profile empty ============\n\r");
		return;
	}
	
	::dumpprint(os,"\n\r=========== Profile ==================\n\r");

	__int64 All = 0;
	int Count = 0;
	CHWProfileEntry* That;
	for(CHWProfileEntry* That = First; That; That = That->Next)
	{              
		if(That->Line >= 0)
		{
			Count++;
    		All += That->xTicks;
		};
    }

	CHWProfileEntry** Map = new CHWProfileEntry*[Count];

	Count = 0;	
	for(That = First; That; That = That->Next)
	{         
		if(That->Line >= 0)
		{	
			int i = 0;
			for (i=0; i<Count && Map[i]->xTicks > That->xTicks; i++);
			for (int j=Count; j>i; j--)
				Map[j] = Map[j-1];
			Map[i] = That;
			Count++;
		};
	};

	__int64 ShowTime = All;
	__int64 TimeNotShown = All;
	int ShowCount = Count;
	if(Hide>0)
		ShowTime -= All/Hide;
	else if(-Hide > 0 && -Hide < Count)
		ShowCount = -Hide;
	int LFileName = 0;

	int i = 0;
	for(i = 0; i < ShowCount && ShowTime > 0; i++)
	{
		That = Map[i];
		__int64 xTicks = That->xTicks;
		if(xTicks < 0) xTicks = 0;
		ShowTime -= xTicks;
		TimeNotShown -= xTicks;
		int iLFileName = (int)::strlen(That->FileName);

		if(LFileName < iLFileName) 
			LFileName = iLFileName;
	};
	ShowCount = i;
	
	for(i = 0; i < ShowCount; i++)
	{
		That = Map[i];
		__int64 xTicks = That->xTicks;
		if(xTicks < 0) xTicks = 0;
		::dumpprint(os,That->FileName);
		::dumpprint(os,"(");
		int LLine = ::dumpprint(os,That->Line);
		::dumpprint(os,"):");
		
		int Spaces = LFileName + 6 - (int)::strlen(That->FileName) - LLine;
		while(Spaces > 0)
		{
			::dumpprint(os," ");
			--Spaces;
		};

		::dumpprint(os,i);
		::dumpprint(os,":\t");
		int LCalls = ::dumpprint(os,That->Calls);
		::dumpprint(os,"x");
		while(LCalls++ < 6)	::dumpprint(os," ");
		::dumpprint(os,"\t");
		::dumpprinttime(os,xTicks, That->Calls);
		::dumpprint(os," \t");
		::dumpprinttime(os,xTicks, 1);
		::dumpprint(os," \t");
		::dumpprint(os,__int64(100.0*float(xTicks) / float(All)));
		::dumpprint(os,"%\t");
		::dumpprint(os,That->Flag);
		::dumpprint(os,"\n\r");
	};

	delete Map;

	::dumpprint(os,"Total: \t");
	::dumpprinttime(os,All, 1);
	
	if(i < Count) 
	{
		::dumpprint(os," (");
		::dumpprint(os,Count-i);
		::dumpprint(os," not-shown-items ");
		::dumpprinttime(os,TimeNotShown,1);
		::dumpprint(os,")");
	};

	::dumpprint(os,"\n\r======================================\n\r");
	
};






