#include "wrapper_search.h"
#include "pb_class.h"

poisson_boltzmann* pb_global_wrapper = nullptr;

int cerca_atomo_wrapper(p8est_t* p4est,
                        p4est_topidx_t which_tree,
                        p8est_quadrant_t* quadrant,
                        p4est_locidx_t local_num,
                        void* point)
{
    if (!pb_global_wrapper)
        return -1;

    return pb_global_wrapper->cerca_atomo(
        p4est,
        which_tree,
        quadrant,
        local_num,
        point   // this is the index of the atom being searched
    );
}