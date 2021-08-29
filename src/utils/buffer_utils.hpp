#ifndef UTILS_BUFFER_UTILS_HPP_
#define UTILS_BUFFER_UTILS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffer_utils.hpp
//! \brief prototypes of utility functions to pack/unpack buffers

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

namespace BufferUtility {
//----------------------------------------------------------------------------------------
//! \fn template <typename T> void PackData(const AthenaArray<T> &src, T *buf,
//!     int sn, int en, int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//! \brief pack a 4D AthenaArray into a one-dimensional buffer

template <typename T> inline void PackData(const AthenaArray<T> &src, T *buf,
         int sn, int en, int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int n=sn; n<=en; ++n) {
    for (int k=sk; k<=ek; k++) {
      for (int j=sj; j<=ej; j++) {
#pragma clang loop vectorize(assume_safety)
        for (int i=si; i<=ei; i++)
          buf[offset++] = src(n,k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void PackData(const AthenaArray<T> &src, T *buf,
//!                     int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//! \brief pack a 3D AthenaArray into a one-dimensional buffer

template <typename T> inline void PackData(const AthenaArray<T> &src, T *buf,
                                    int si, int ei, int sj, int ej, int sk, int ek,
                                    int &offset) {
  for (int k=sk; k<=ek; k++) {
    for (int j=sj; j<=ej; j++) {
#pragma clang loop vectorize(assume_safety)
      for (int i=si; i<=ei; i++)
        buf[offset++] = src(k, j, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackData(const T *buf, AthenaArray<T> &dst,
//!     int sn, int en, int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//! \brief unpack a one-dimensional buffer into a 4D AthenaArray

template <typename T> inline void UnpackData(const T *buf, AthenaArray<T> &dst,
         int sn, int en, int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int n=sn; n<=en; ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma clang loop vectorize(assume_safety)
        for (int i=si; i<=ei; ++i)
          dst(n,k,j,i) = buf[offset++];
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackData(const T *buf, AthenaArray<T> &dst,
//!                       int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//! \brief unpack a one-dimensional buffer into a 3D AthenaArray

template <typename T> inline void UnpackData(const T *buf, AthenaArray<T> &dst,
                           int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma clang loop vectorize(assume_safety)
      for (int i=si; i<=ei; ++i)
        dst(k,j,i) = buf[offset++];
    }
  }
  return;
}
} // namespace BufferUtility
#endif // UTILS_BUFFER_UTILS_HPP_
