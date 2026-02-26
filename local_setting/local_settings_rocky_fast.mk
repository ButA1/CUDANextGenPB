CPPFLAGS=\
-I/usr/lib64/openmpi/include \
-I/opt/openmpi/include \
-I/usr/include/MUMPS/ \
-I/opt/octave_file_io/include/ \
-I/opt/p4est/include/ \
-I/opt/nanoshaper/src \
-I/usr/local/nextgenPB/include -I/usr/local/nextgenPB/addons \
-I/opt/bimpp/include \
-I/opt/lis/include \
-DHAVE_OCTAVE_44 -DOMPI_SKIP_MPICXX -DBIM_TIMING -DUSE_MPI

CXXFLAGS= -Ofast -mtune=native -march=native
#CXXFLAGS= -O2 -mtune=generic
CXX=mpicxx

LDFLAGS=-L/opt/octave_file_io/lib \
-L/opt/p4est/lib \
-L/opt/lis/lib \
-L/opt/nanoshaper/build_lib \
-L/opt/bimpp/lib \
-L/opt/p4est/lib \
-L/usr/lib64/openmpi/lib \
-L/opt/openmpi/lib \
-Wl,-rpath,/opt/nanoshaper/build_lib \
-Wl,-rpath,/opt/bimpp/lib  \
-Wl,-rpath,/opt/octave_file_io/lib \
-Wl,-rpath,/opt/lis/lib \
-Wl,-rpath,/opt/p4est/lib

LIBS=-lNanoShaper -lbim -lbimmumps -lbimlis -lbimp4est \
     -lbimlinalg -llis -ldmumps -lmumps_common \
     -lscotcherr -lbz2 -lmpi_usempif08 \
     -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lp4est -lsc
