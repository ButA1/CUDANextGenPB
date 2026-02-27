CXX=mpicxx

NVCC=nvcc

CXXFLAGS= -std=c++17 -ggdb -O0

NVCCFLAGS= -arch=native

CPPFLAGS=-I/opt/octave_file_io/1.0.91/include/ \
-I/opt/nanoshaper/src \
-I../include -I../addons \
-I/opt/bimpp/include \
-I/opt/local/include \
-DHAVE_OCTAVE_44 -DOMPI_SKIP_MPICXX -DBIM_TIMING -DUSE_MPI

LDFLAGS=-L/opt/octave_file_io/1.0.91/lib \
-L/opt/local/lib \
-L/opt/nanoshaper/build_lib\
-L/opt/bimpp_ngpb/lib \
-Wl,-rpath,/opt/nanoshaper/build_lib \
-Wl,-rpath,/opt/bimpp_ngpb/lib  \
-Wl,-rpath,/opt/octave_file_io/1.0.91/lib

LIBS=-lNanoShaper -lbim -lbimmumps -lbimlis -lbimp4est -lbimlinalg -llis -ldmumps -lmumps_common \
-lscotcherr -lbz2 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lp4est -lsc

include local_settings.mk

all : ngpb

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $<

fast:
	$(NVCC) $(NVCCFLAGS) -c fast.cu -o fast.o

ngpb : $(patsubst %.cpp, %.o, $(wildcard *.cpp)) fast.o
	$(NVCC) -dlink $(patsubst %.cpp, %.o, $(wildcard *.cpp)) fast.o -o device_link.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(patsubst %.cpp, %.o, $(wildcard *.cpp)) fast.o device_link.o -o $@ $(LIBS)

.PHONY : clean distclean fast

clean :
	$(RM) *.o

distclean : clean
	$(RM) ngpb device_link.o


