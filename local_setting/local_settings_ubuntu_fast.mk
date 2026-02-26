CPPFLAGS=\
-I/usr/include/ -I/usr/include/octave-8.4.0/octave/8.4.0/ \
-I/opt/octave_file_io/include/ \
-I/opt/nanoshaper/src \
-I/usr/local/nextgenPB/include -I/usr/local/nextgenPB/addons \
-I/opt/bimpp/include \
-I/opt/lis/include \
$(shell mkoctfile --link-stand-alone -p INCFLAGS) \
-DHAVE_OCTAVE_44 -DOMPI_SKIP_MPICXX -DBIM_TIMING -DUSE_MPI


#CXXFLAGS= -O2 -mtune=generic
CXXFLAGS= -Ofast -mtune=native -march=native
CXX=mpicxx

LDFLAGS=-L/opt/octave_file_io/lib \
-L/usr/lib/x86_64-linux-gnu \
-L/usr/lib/x86_64-linux-gnu/octave/8.4.0/ \
-L/opt/lis/lib \
-L/opt/nanoshaper/build_lib \
-L/opt/bimpp/lib \
$(shell mkoctfile --link-stand-alone -p RDYNAMIC_FLAG) \
$(shell mkoctfile --link-stand-alone -p LDFLAGS) \
-Wl,-rpath,/opt/nanoshaper/build_lib \
-Wl,-rpath,/opt/bimpp/lib  \
-Wl,-rpath,/opt/octave_file_io/lib \
-Wl,-rpath,/opt/lis/lib 

LIBS=-lNanoShaper -lbim -lbimmumps -lbimlis -lbimp4est \
     -lbimlinalg -loctave_file_io \
     $(shell mkoctfile --link-stand-alone -p OCTAVE_LIBS) \
     -llis -ldmumps -lmumps_common \
     -lscotcherr -lbz2 -lmpi_usempif08 \
     -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lp4est -lsc
