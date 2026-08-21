/* Minimal MPI transport benchmark: tells you whether the container's OpenMPI is
 * using shared memory / RDMA or has silently fallen back to TCP.
 *
 * Build (inside the container):
 *   singularity exec ngpb.sif mpicc -O2 -o mpi_bench mpi_bench.c
 *
 * Reference numbers on a modern cluster:
 *   intra-node latency   0.2 - 0.5 us   (vader/CMA)     vs  15 - 50 us if on TCP
 *   inter-node latency   1 - 3 us       (IB/UCX)        vs  20 - 60 us if on TCP
 *   bandwidth            10+ GB/s shm, 10 - 25 GB/s IB  vs  0.1 - 1 GB/s on GbE
 *   allreduce(8B) is what p8est_refine hammers -- it tracks latency, not bandwidth.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WARM 100
#define ITER 2000
#define BIGSZ (8 * 1024 * 1024)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  char host[256];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  gethostname(host, sizeof host);

  for (int r = 0; r < size; ++r) {
    if (r == rank) printf("rank %d on %s\n", rank, host);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (size < 2) { if (!rank) printf("need >= 2 ranks\n"); MPI_Finalize(); return 0; }

  /* --- ping-pong latency, rank 0 <-> rank 1 --- */
  char small[8];
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank < 2) {
    for (int i = 0; i < WARM; ++i) {
      if (rank == 0) { MPI_Send(small,8,MPI_CHAR,1,0,MPI_COMM_WORLD);
                       MPI_Recv(small,8,MPI_CHAR,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
      else           { MPI_Recv(small,8,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                       MPI_Send(small,8,MPI_CHAR,0,0,MPI_COMM_WORLD); }
    }
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITER; ++i) {
      if (rank == 0) { MPI_Send(small,8,MPI_CHAR,1,0,MPI_COMM_WORLD);
                       MPI_Recv(small,8,MPI_CHAR,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
      else           { MPI_Recv(small,8,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                       MPI_Send(small,8,MPI_CHAR,0,0,MPI_COMM_WORLD); }
    }
    double t1 = MPI_Wtime();
    if (rank == 0)
      printf("\nping-pong latency (8B, rank0<->rank1): %8.2f us\n",
             (t1 - t0) * 1e6 / (2.0 * ITER));
  }

  /* --- bandwidth, rank 0 <-> rank 1 --- */
  char *big = malloc(BIGSZ);
  memset(big, 1, BIGSZ);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank < 2) {
    double t0 = MPI_Wtime();
    for (int i = 0; i < 20; ++i) {
      if (rank == 0) { MPI_Send(big,BIGSZ,MPI_CHAR,1,1,MPI_COMM_WORLD);
                       MPI_Recv(big,BIGSZ,MPI_CHAR,1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
      else           { MPI_Recv(big,BIGSZ,MPI_CHAR,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                       MPI_Send(big,BIGSZ,MPI_CHAR,0,1,MPI_COMM_WORLD); }
    }
    double t1 = MPI_Wtime();
    if (rank == 0)
      printf("bandwidth (8MB pairs):                 %8.2f GB/s\n",
             (2.0 * 20 * BIGSZ) / (t1 - t0) / 1e9);
  }
  free(big);

  /* --- small allreduce over ALL ranks: this is the p8est_refine pattern --- */
  double v = 1.0, out;
  for (int i = 0; i < WARM; ++i) MPI_Allreduce(&v,&out,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < ITER; ++i) MPI_Allreduce(&v,&out,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  if (rank == 0)
    printf("allreduce (8B, %d ranks):               %8.2f us   <-- p8est pattern\n",
           size, (t1 - t0) * 1e6 / ITER);

  MPI_Finalize();
  return 0;
}
