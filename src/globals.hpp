#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file globals.hpp
//! \brief namespace containing external global variables

namespace Globals {
extern int my_rank, nranks;
}

#ifdef UTOFU_PARALLEL
#include <utofu.h>
namespace Utofu {
extern size_t num_tnis;
extern utofu_tni_id_t *tni_ids;
}
#endif

#endif // GLOBALS_HPP_
