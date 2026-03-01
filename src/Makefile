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
-L/lib/x86_64-linux-gnu \
-Wl,-rpath,/opt/nanoshaper/build_lib \
-Wl,-rpath,/opt/bimpp_ngpb/lib  \
-Wl,-rpath,/opt/octave_file_io/1.0.91/lib \
-Wl,-rpath,/lib/x86_64-linux-gnu

LIBS=-lNanoShaper -lbim -lbimmumps -lbimlis -lbimp4est -lbimlinalg -llis -ldmumps -lmumps_common \
-lscotcherr -lbz2 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lp4est -lsc -lcudss

include local_settings.mk

all : ngpb

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $<

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

CUDA_SOURCES = fast.cu simple.cu
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

cuda: $(CUDA_OBJECTS)

ngpb : $(patsubst %.cpp, %.o, $(wildcard *.cpp)) $(CUDA_OBJECTS)
	$(NVCC) -dlink $(patsubst %.cpp, %.o, $(wildcard *.cpp)) $(CUDA_OBJECTS) -o device_link.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(patsubst %.cpp, %.o, $(wildcard *.cpp)) $(CUDA_OBJECTS) device_link.o -o $@ $(LIBS)

.PHONY : clean distclean cuda

clean :
	$(RM) *.o

distclean : clean
	$(RM) ngpb device_link.o


