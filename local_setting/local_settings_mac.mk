CPPFLAGS=\
-I/opt/octave_file_io/1.0.91/include/ \
-I/opt/nanoshaper/src \
-I../include -I../addons \
-I/opt/bimpp/include \
-I/opt/local/include \
-DHAVE_OCTAVE_44 -DOMPI_SKIP_MPICXX -DBIM_TIMING -DUSE_MPI

#CXXFLAGS= -std=c++17 -ggdb  -O2 -mtune=generic
CXXFLAGS= -std=c++17 -Ofast -mtune=native -march=native

LDFLAGS=-L/opt/octave_file_io/1.0.91/lib \
-L/opt/local/lib \
-L/opt/nanoshaper/build_lib\
-L/opt/bimpp/lib \
-Wl,-rpath,/opt/nanoshaper/build_lib \
-Wl,-rpath,/opt/bimpp_ngpb/lib  \
-Wl,-rpath,/opt/octave_file_io/1.0.91/lib

LIBS=-lNanoShaper -lbim -lbimmumps -lbimlis -lbimp4est -lbimlinalg -llis -ldmumps -lmumps_common \
-lscotcherr -lbz2 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lp4est -lsc
