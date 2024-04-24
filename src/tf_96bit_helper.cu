/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.barrett

You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/


__device__ static void check_factor96(int96 f, int96 a, bool negativeBase, unsigned int *RES)
/* Check whether f is a factor or not. If f != 1 and a == 1 then f is a factor,
in this case f is written into the RES array. */
{
  unsigned int index;
  bool isFactor;
  if (negativeBase)
  {
    isFactor = a.d2 == f.d2 && a.d1 == f.d1 && a.d0 == (f.d0 - 1);
  } else
  {
    isFactor = (a.d2|a.d1) == 0 && a.d0 == 1;
  }
  if(isFactor)
  {
    if(f.d2 != 0 || f.d1 != 0 || f.d0 != 1)	/* 1 isn't really a factor ;) */
    {
      index=atomicInc(&RES[0], 10000);
      if(index < 10)				/* limit to 10 factors per class */
      {
        RES[index * 3 + 1] = f.d2;
        RES[index * 3 + 2] = f.d1;
        RES[index * 3 + 3] = f.d0;
      }
    }
  }
}


__device__ static void create_FC96(int96 *f, unsigned int exp, int96 k, unsigned int k_offset)
/* calculates f = 2 * (k+k_offset) * exp + 1 */
{
  int96 exp96;

  exp96.d1 = exp >> 31;
  exp96.d0 = exp << 1;			// exp96 = 2 * exp

  k.d0 = __add_cc (k.d0, __umul32  (k_offset, NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_offset, NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */

  f->d0 = 1 +                                  __umul32(k.d0, exp96.d0);
  f->d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f->d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f->d1 = __add_cc(f->d1, k.d0);
    f->d2 = __addc  (f->d2, k.d1);
  }							// f = 2 * k * exp + 1
}


__device__ static void create_FC96_mad(int96 *f, unsigned int exp, int96 k, unsigned int k_offset)
/* similar to create_FC96(), this versions uses multiply-add with carry which
is faster for _SOME_ kernels. */
{
#if (__CUDA_ARCH__ < FERMI) || (CUDART_VERSION < 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  create_FC96(f, exp, k, k_offset);
#else
  int96 exp96;

  exp96.d1 = exp >> 31;
  exp96.d0 = exp << 1;			// exp96 = 2 * exp

  k.d0 = __umad32_cc(k_offset, NUM_CLASSES, k.d0);
  k.d1 = __umad32hic(k_offset, NUM_CLASSES, k.d1);

  /* umad32 is slower here?! */
  f->d0 = 1 +                                  __umul32(k.d0, exp96.d0);
  f->d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f->d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f->d1 = __add_cc(f->d1, k.d0);
    f->d2 = __addc  (f->d2, k.d1);
  }							// f = 2 * k * exp + 1
#endif
}

