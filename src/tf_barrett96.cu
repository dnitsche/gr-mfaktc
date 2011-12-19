/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                
You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>  

#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "my_intrinsics.h"

#define NVCC_EXTERN
#include "sieve.h"
#include "timer.h"
#undef NVCC_EXTERN

#include "tf_debug.h"


extern __host__ void print_dez96(int96 a, char *buf);


__device__ static int cmp_ge_96(int96 a, int96 b)
/* checks if a is greater or equal than b */
{
  if(a.d2 == b.d2)
  {
    if(a.d1 == b.d1)return(a.d0 >= b.d0);
    else            return(a.d1 >  b.d1);
  }
  else              return(a.d2 >  b.d2);
}


__device__ static void sub_96(int96 *res, int96 a, int96 b)
/* a must be greater or equal b!
res = a - b */
{
  res->d0 = __sub_cc (a.d0, b.d0);
  res->d1 = __subc_cc(a.d1, b.d1);
  res->d2 = __subc   (a.d2, b.d2);
}


__device__ static void mul_96(int96 *res, int96 a, int96 b)
/* res = a * b (only lower 96 bits of the result) */
{
#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      ".reg .u32 t1;\n\t"

      "mul.lo.u32    %0, %3, %6;\n\t"       /* (a.d0 * b.d0).lo */

      "mul.hi.u32    t1, %3, %6;\n\t"       /* (a.d0 * b.d0).hi */
      "mad.lo.cc.u32 %1, %4, %6, t1;\n\t"   /* (a.d1 * b.d0).lo */
      "mul.lo.u32    t1, %5, %6;\n\t"       /* (a.d2 * b.d0).lo */
      "madc.hi.u32   %2, %4, %6, t1;\n\t"   /* (a.d1 * b.d0).hi */

      "mad.lo.cc.u32 %1, %3, %7, %1;\n\t"   /* (a.d0 * b.d1).lo */
      "madc.hi.u32   %2, %3, %7, %2;\n\t"   /* (a.d0 * b.d1).hi */

      "mul.lo.u32    t1, %3, %8;\n\t"       /* (a.d0 * b.d2).lo */
      "add.u32       %2, %2, t1;\n\t"

      "mul.lo.u32    t1, %4, %7;\n\t"       /* (a.d1 * b.d1).lo */
      "add.u32       %2, %2, t1;\n\t"
      "}"
      : "=r" (res->d0), "=r" (res->d1), "=r" (res->d2)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2), "r" (b.d0), "r" (b.d1), "r" (b.d2));
#else
  res->d0 = __umul32  (a.d0, b.d0);

  res->d1 = __add_cc(__umul32hi(a.d0, b.d0), __umul32  (a.d1, b.d0));
  res->d2 = __addc  (__umul32  (a.d2, b.d0), __umul32hi(a.d1, b.d0));
  
  res->d1 = __add_cc(res->d1,                __umul32  (a.d0, b.d1));
  res->d2 = __addc  (res->d2,                __umul32hi(a.d0, b.d1));

  res->d2+= __umul32  (a.d0, b.d2);

  res->d2+= __umul32  (a.d1, b.d1);
#endif
}


//__device__ static void mul_96_192(int192 *res, int96 a, int96 b)
/* res = a * b */
/*{
  res->d0 = __umul32  (a.d0, b.d0);
  res->d1 = __umul32hi(a.d0, b.d0);
  res->d2 = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d1 = __add_cc (res->d1, __umul32  (a.d1, b.d0));
  res->d2 = __addc_cc(res->d2, __umul32hi(a.d1, b.d0));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d1 = __add_cc (res->d1, __umul32  (a.d0, b.d1));
  res->d2 = __addc_cc(res->d2, __umul32hi(a.d0, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
  res->d5 = __addc   (      0,                      0);

  res->d2 = __add_cc (res->d2, __umul32  (a.d0, b.d2));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));

  res->d2 = __add_cc (res->d2, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
}*/


__device__ static void mul_96_192_no_low2(int192 *res, int96 a, int96 b)
/*
res ~= a * b
res.d0 and res.d1 are NOT computed. Carry from res.d1 to res.d2 is ignored,
too. So the digits res.d{2-5} might differ from mul_96_192(). In
mul_96_192() are two carries from res.d1 to res.d2. So ignoring the digits
res.d0 and res.d1 the result of mul_96_192_no_low() is 0 to 2 lower than
of mul_96_192().
 */
{
#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      "mul.lo.u32      %0, %6, %7;\n\t"       /* (a.d2 * b.d0).lo */
      "mul.hi.u32      %1, %6, %7;\n\t"       /* (a.d2 * b.d0).hi */

      "mad.hi.cc.u32   %0, %5, %7, %0;\n\t"   /* (a.d1 * b.d0).hi */
      "madc.lo.cc.u32  %1, %6, %8, %1;\n\t"   /* (a.d2 * b.d1).lo */
      "addc.u32        %2,  0,  0;\n\t"

      "mad.hi.cc.u32   %0, %4, %8, %0;\n\t"   /* (a.d0 * b.d1).hi */
      "madc.lo.cc.u32  %1, %5, %9, %1;\n\t"   /* (a.d1 * b.d2).lo */
      "madc.hi.cc.u32  %2, %5, %9, %2;\n\t"   /* (a.d1 * b.d2).hi */
      "addc.u32        %3,  0,  0;\n\t"

      "mad.lo.cc.u32   %0, %4, %9, %0;\n\t"   /* (a.d0 * b.d2).lo */
      "madc.hi.cc.u32  %1, %4, %9, %1;\n\t"   /* (a.d0 * b.d2).hi */
      "madc.lo.cc.u32  %2, %6, %9, %2;\n\t"   /* (a.d2 * b.d2).lo */
      "madc.hi.u32     %3, %6, %9, %3;\n\t"   /* (a.d2 * b.d2).hi */

      "mad.lo.cc.u32   %0, %5, %8, %0;\n\t"   /* (a.d1 * b.d1).lo */
      "madc.hi.cc.u32  %1, %5, %8, %1;\n\t"   /* (a.d1 * b.d1).hi */
      "madc.hi.cc.u32  %2, %6, %8, %2;\n\t"   /* (a.d2 * b.d1).lo */
      "addc.u32        %3, %3,  0;\n\t"
      "}"
      : "=r" (res->d2), "=r" (res->d3), "=r" (res->d4), "=r" (res->d5)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2), "r" (b.d0), "r" (b.d1), "r" (b.d2));
#else
  res->d2 = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d2 = __add_cc (res->d2, __umul32hi(a.d1, b.d0));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d2 = __add_cc (res->d2, __umul32hi(a.d0, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
  res->d5 = __addc   (      0,                      0);

  res->d2 = __add_cc (res->d2, __umul32  (a.d0, b.d2));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));

  res->d2 = __add_cc (res->d2, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
#endif
}


__device__ static void mul_96_192_no_low3(int192 *res, int96 a, int96 b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is ignored,
too. So the digits res.d{3-5} might differ from mul_96_192(). In
mul_96_192() are four carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_96_192_no_low() is 0 to 4 lower
than of mul_96_192().
 */
{
#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      "mul.hi.u32      %0, %5, %6;\n\t"       /* (a.d2 * b.d0).hi */
      "mad.lo.cc.u32   %0, %5, %7, %0;\n\t"   /* (a.d2 * b.d1).lo */
      "addc.u32        %1,  0,  0;\n\t"

      "mad.lo.cc.u32   %0, %4, %8, %0;\n\t"   /* (a.d1 * b.d2).lo */
      "madc.hi.u32     %1, %4, %8, %1;\n\t"   /* (a.d1 * b.d2).hi */

      "mad.hi.cc.u32   %0, %3, %8, %0;\n\t"   /* (a.d0 * b.d2).hi */
      "madc.lo.cc.u32  %1, %5, %8, %1;\n\t"   /* (a.d2 * b.d2).lo */
      "madc.hi.u32     %2, %5, %8,  0;\n\t"   /* (a.d2 * b.d2).hi */

      "mad.hi.cc.u32   %0, %4, %7, %0;\n\t"   /* (a.d1 * b.d1).hi */
      "madc.hi.cc.u32  %1, %5, %7, %1;\n\t"   /* (a.d2 * b.d1).lo */
      "addc.u32        %2, %2,  0;\n\t"
      "}"
      : "=r" (res->d3), "=r" (res->d4), "=r" (res->d5)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2), "r" (b.d0), "r" (b.d1), "r" (b.d2));
#else
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc   (res->d4, __umul32hi(a.d1, b.d2)); // no carry propagation to d5 needed: 0xFFFF.FFFF * 0xFFFF.FFFF + 0xFFFF.FFFF + 0xFFFF.FFFE = 0xFFFF.FFFF.FFFF.FFFE
//  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
//  res->d5 = __addc   (      0,                      0);

  res->d3 = __add_cc (res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
//  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));
  res->d5 = __addc   (      0, __umul32hi(a.d2, b.d2));

  res->d3 = __add_cc (res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
#endif
}


__device__ static void square_96_192(int192 *res, int96 a)
/* res = a^2
assuming that a is < 2^95 (a.d2 < 2^31)! */
{
#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      ".reg .u32 a2, t1, t2;\n\t"

      "mul.lo.u32      %0, %6, %6;\n\t"       /* (a.d0 * a.d0).lo */
      "mul.hi.u32      %1, %6, %6;\n\t"       /* (a.d0 * a.d0).hi */
      "mul.lo.u32      %2, %7, %7;\n\t"       /* (a.d1 * a.d1).lo */
      "mul.hi.u32      %3, %7, %7;\n\t"       /* (a.d1 * a.d1).hi */
      "mul.lo.u32      %4, %8, %8;\n\t"       /* (a.d2 * a.d2).lo */
/* highest possible value for __umul32  (N, N) is 0xFFFFFFF9
this occurs for N = {479772853, 1667710795, 2627256501, 3815194443}
We'll use this knowledge later to avoid some two carry steps to %5 */

      "add.u32         a2, %8, %8;\n\t"       /* a2 = 2 * a.d2 */
      
      "mul.lo.u32      t1, %6, %7;\n\t"
      "mul.hi.u32      t2, %6, %7;\n\t"

      "add.cc.u32      %1, %1, t1;\n\t"       /* (a.d0 * a.d1).lo */
      "addc.cc.u32     %2, %2, t2;\n\t"       /* (a.d0 * a.d1).hi */
      "madc.hi.cc.u32  %3, %6, a2, %3;\n\t"   /* (a.d0 * a.d2).hi + (a.d2 * a.d0).hi */
      "addc.u32        %4, %4,  0;\n\t"       /* %4 <= 0xFFFFFFFA => not carry to %5 needed, see above! */

      "add.cc.u32      %1, %1, t1;\n\t"       /* (a.d1 * a.d0).lo */
      "addc.cc.u32     %2, %2, t2;\n\t"       /* (a.d1 * a.d0).hi */
      "addc.cc.u32     %3, %3,  0;\n\t"
      "addc.u32        %4, %4,  0;\n\t"       /* %4 <= 0xFFFFFFFB => not carry to %5 needed, see above! */
      
      "mad.lo.cc.u32   %2, %6, a2, %2;\n\t"   /* (a.d0 * a.d2).lo + (a.d2 * a.d0).lo */
      "madc.lo.cc.u32  %3, %7, a2, %3;\n\t"   /* (a.d1 * a.d2).lo + (a.d2 * a.d1).lo */
      "madc.hi.cc.u32  %4, %7, a2, %4;\n\t"   /* (a.d1 * a.d2).hi + (a.d2 * a.d1).hi */
      "madc.hi.u32     %5, %8, %8,  0;\n\t"   /* (a.d2 * a.d2).hi */
      "}"
      : "=r" (res->d0), "=r" (res->d1), "=r" (res->d2), "=r" (res->d3), "=r" (res->d4), "=r" (res->d5)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2));
#else
  asm("{\n\t"
      ".reg .u32 a2, t1, t2, t3;\n\t"

      "mul.lo.u32      %0, %6, %6;\n\t"       /* (a.d0 * a.d0).lo */
      "mul.hi.u32      %1, %6, %6;\n\t"       /* (a.d0 * a.d0).hi */
      "mul.lo.u32      %2, %7, %7;\n\t"       /* (a.d1 * a.d1).lo */
      "mul.hi.u32      %3, %7, %7;\n\t"       /* (a.d1 * a.d1).hi */
      "mul.lo.u32      %4, %8, %8;\n\t"       /* (a.d2 * a.d2).lo */
/* highest possible value for __umul32  (N, N) is 0xFFFFFFF9
this occurs for N = {479772853, 1667710795, 2627256501, 3815194443}
We'll use this knowledge later to avoid some two carry steps to %5 */

      "add.u32         a2, %8, %8;\n\t"       /* a2 = 2 * a.d2 */
      
      "mul.lo.u32      t1, %6, %7;\n\t"
      "mul.hi.u32      t2, %6, %7;\n\t"

      "add.cc.u32      %1, %1, t1;\n\t"       /* (a.d0 * a.d1).lo */
      "addc.cc.u32     %2, %2, t2;\n\t"       /* (a.d0 * a.d1).hi */
      "mul.hi.u32      t3, %6, a2;\n\t"       /* (a.d0 * a.d2).hi + (a.d2 * a.d0).hi */
      "addc.cc.u32     %3, %3, t3;\n\t"
      "addc.u32        %4, %4,  0;\n\t"       /* %4 <= 0xFFFFFFFA => not carry to %5 needed, see above! */

      "add.cc.u32      %1, %1, t1;\n\t"       /* (a.d1 * a.d0).lo */
      "addc.cc.u32     %2, %2, t2;\n\t"       /* (a.d1 * a.d0).hi */
      "addc.cc.u32     %3, %3,  0;\n\t"
      "addc.u32        %4, %4,  0;\n\t"       /* %4 <= 0xFFFFFFFB => not carry to %5 needed, see above! */
      
      "mul.lo.u32      t3, %6, a2;\n\t"       /* (a.d0 * a.d2).lo + (a.d2 * a.d0).lo */
      "add.cc.u32      %2, %2, t3;\n\t"
      "mul.lo.u32      t3, %7, a2;\n\t"       /* (a.d1 * a.d2).lo + (a.d2 * a.d1).lo */
      "addc.cc.u32     %3, %3, t3;\n\t"
      "mul.hi.u32      t3, %7, a2;\n\t"       /* (a.d1 * a.d2).hi + (a.d2 * a.d1).hi */
      "addc.cc.u32     %4, %4, t3;\n\t"
      "mul.hi.u32      t3, %8, %8;\n\t"       /* (a.d2 * a.d2).hi */
      "addc.u32        %5, t3,  0;\n\t"
      "}"
      : "=r" (res->d0), "=r" (res->d1), "=r" (res->d2), "=r" (res->d3), "=r" (res->d4), "=r" (res->d5)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2));
#endif
}


__device__ static void square_96_160(int192 *res, int96 a)
/* res = a^2
this is a stripped down version of square_96_192, it doesn't compute res.d5
and is a little bit faster.
For correct results a must be less than 2^80 (a.d2 less than 2^16) */
{
#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      ".reg .u32 t1, t2, t3;\n\t"

      "mul.lo.u32     %0, %5, %5;\n\t"     /* (a.d0 * a.d0).lo */
      "mul.hi.u32     %1, %5, %5;\n\t"     /* (a.d0 * a.d0).hi */

      "mul.lo.u32     %4, %7, %7;\n\t"     /* (a.d2 * a.d2).lo */

      "add.u32        t3, %7, %7;\n\t"     /* shl(a.d2) */

      "mul.lo.u32     %2, %5, t3;\n\t"     /* 2(a.d0 * a.d2).lo */
      "mul.hi.u32     %3, %5, t3;\n\t"     /* 2(a.d0 * a.d2).hi */

      "mul.lo.u32     t1, %5, %6;\n\t"     /* (a.d0 * a.d1).lo */
      "add.cc.u32     %1, %1, t1;\n\t"
      "mul.hi.u32     t2, %5, %6;\n\t"     /* (a.d0 * a.d1).hi */
      "addc.cc.u32    %2, %2, t2;\n\t"
      "addc.u32       %3, %3,  0;\n\t"     /* %3 (res.d3) has some space left because a.d2 is < 2^16 */

      "add.cc.u32     %1, %1, t1;\n\t"
      "addc.cc.u32    %2, %2, t2;\n\t"
      "madc.lo.cc.u32 %3, %6, t3, %3;\n\t" /* 2(a.d1 * a.d2).lo */
      "madc.hi.u32    %4, %6, t3, %4;\n\t" /* 2(a.d1 * a.d2).hi */

      "mad.lo.cc.u32  %2, %6, %6, %2;\n\t" /* (a.d1 * a.d1).lo */
      "madc.hi.cc.u32 %3, %6, %6, %3;\n\t" /* (a.d1 * a.d1).hi */
      "addc.u32       %4, %4,  0;\n\t"
      "}"
      : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3), "=r"(res->d4)
      : "r"(a.d0), "r"(a.d1), "r"(a.d2));
#else
  asm("{\n\t"
      ".reg .u32 t1, t2, t3, t4;\n\t"

      "mul.lo.u32    %0, %5, %5;\n\t" // (a.d0 * a.d0).lo
      "mul.hi.u32    %1, %5, %5;\n\t" // (a.d0 * a.d0).hi

      "mul.lo.u32    %4, %7, %7;\n\t" // (a.d2 * a.d2).lo
      
      "add.u32       t4, %7, %7;\n\t" // shl(a.d2)
      
      "mul.lo.u32    %2, %5, t4;\n\t" // 2(a.d0 * a.d2).lo
      "mul.hi.u32    %3, %5, t4;\n\t" // 2(a.d0 * a.d2).hi
      
      "mul.lo.u32    t1, %5, %6;\n\t" // (a.d0 * a.d1).lo
      "add.cc.u32    %1, %1, t1;\n\t"
      "mul.hi.u32    t2, %5, %6;\n\t" // (a.d0 * a.d1).hi
      "addc.cc.u32   %2, %2, t2;\n\t"
      "addc.u32      %3, %3,  0;\n\t" // %3 (res.d3) has some space left because a.d2 is < 2^16
      
      "add.cc.u32    %1, %1, t1;\n\t"
      "addc.cc.u32   %2, %2, t2;\n\t"
      "mul.lo.u32    t3, %6, t4;\n\t" // 2(a.d1 * a.d2).lo
      "addc.cc.u32   %3, %3, t3;\n\t"
      "mul.hi.u32    t3, %6, t4;\n\t" // 2(a.d1 * a.d2).hi
      "addc.u32      %4, %4, t3;\n\t"
      
      "mul.lo.u32    t1, %6, %6;\n\t" // (a.d1 * a.d1).lo
      "add.cc.u32    %2, %2, t1;\n\t"
      "mul.hi.u32    t1, %6, %6;\n\t" // (a.d1 * a.d1).hi
      "addc.cc.u32   %3, %3, t1;\n\t"
      "addc.u32      %4, %4,  0;\n\t"
      "}"
      : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3), "=r"(res->d4)
      : "r"(a.d0), "r"(a.d1), "r"(a.d2));
#endif 
}


__device__ static void shl_96(int96 *a)
/* shiftleft a one bit */
{
  a->d0 = __add_cc (a->d0, a->d0);
  a->d1 = __addc_cc(a->d1, a->d1);
  a->d2 = __addc   (a->d2, a->d2);
}


#undef DIV_160_96
#ifndef CHECKS_MODBASECASE
__device__ static void div_192_96(int96 *res, int192 q, int96 n, float nf)
#else
__device__ static void div_192_96(int96 *res, int192 q, int96 n, float nf, unsigned int *modbasecase_debug)
#endif
/* res = q / n (integer division) */
{
  float qf;
  unsigned int qi;
  int192 nn;
  int96 tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= __uint2float_rn(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;

// nn = n * qi
  nn.d2 =                                 __umul32(n.d0, qi);
  nn.d3 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
#ifndef DIV_160_96
  nn.d4 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d5 = __addc   (__umul32hi(n.d2, qi),                  0);
#else
  nn.d4 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#endif

// shiftleft nn 11 bits
#ifndef DIV_160_96
  nn.d5 = (nn.d5 << 11) + (nn.d4 >> 21);
#endif
  nn.d4 = (nn.d4 << 11) + (nn.d3 >> 21);
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
  nn.d2 =  nn.d2 << 11;

//  q = q - nn
  q.d2 = __sub_cc (q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc_cc(q.d4, nn.d4);
#ifndef DIV_160_96
  q.d5 = __subc   (q.d5, nn.d5);
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  qf= __uint2float_rn(q.d4);
#endif
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf*= 512.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  qi << 23;
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 23 bits
#ifdef CHECKS_MODBASECASE
  nn.d5 =                  nn.d4 >> 9;
#endif  
  nn.d4 = (nn.d4 << 23) + (nn.d3 >> 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
  nn.d1 =  nn.d1 << 23;

// q = q - nn
  q.d1 = __sub_cc (q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
#ifndef CHECKS_MODBASECASE  
  q.d4 = __subc   (q.d4, nn.d4);
#else
  q.d4 = __subc_cc(q.d4, nn.d4);
  q.d5 = __subc   (q.d5, nn.d5);
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  res->d1 = __add_cc(res->d1, qi << 3 );
  res->d2 = __addc  (res->d2, qi >> 29);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

//  q = q - nn
  q.d1 = __sub_cc (q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc   (q.d4, nn.d4);

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf*= 131072.0f;
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  res->d0 = qi << 15;
  res->d1 = __add_cc(res->d1, qi >> 17);
  res->d2 = __addc  (res->d2, 0);
  
// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
#ifndef CHECKS_MODBASECASE
  q.d3 = __subc   (q.d3, nn.d3);
#else
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc   (q.d4, nn.d4);
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d1);
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 = __add_cc (res->d0, qi);
  res->d1 = __addc_cc(res->d1,  0);
  res->d2 = __addc   (res->d2,  0);
  

// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
#ifndef CHECKS_MODBASECASE  
  nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#else
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);
#endif  

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
#ifndef CHECKS_MODBASECASE
  q.d2 = __subc   (q.d2, nn.d2);
#else
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc   (q.d3, nn.d3);
#endif

//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;
  tmp96.d0=q.d0;
  tmp96.d1=q.d1;
  tmp96.d2=q.d2;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
  MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
  MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
  if(cmp_ge_96(tmp96,n))
  {
    res->d0 = __add_cc (res->d0,  1);
    res->d1 = __addc_cc(res->d1,  0);
    res->d2 = __addc   (res->d2,  0);
  }
}


#define DIV_160_96
#ifndef CHECKS_MODBASECASE
__device__ static void div_160_96(int96 *res, int192 q, int96 n, float nf)
#else
__device__ static void div_160_96(int96 *res, int192 q, int96 n, float nf, unsigned int *modbasecase_debug)
#endif
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  float qf;
  unsigned int qi;
  int192 nn;
  int96 tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= __uint2float_rn(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;

// nn = n * qi
  nn.d2 =                                 __umul32(n.d0, qi);
  nn.d3 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
#ifndef DIV_160_96
  nn.d4 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d5 = __addc   (__umul32hi(n.d2, qi),                  0);
#else
  nn.d4 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#endif

// shiftleft nn 11 bits
#ifndef DIV_160_96
  nn.d5 = (nn.d5 << 11) + (nn.d4 >> 21);
#endif
  nn.d4 = (nn.d4 << 11) + (nn.d3 >> 21);
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
  nn.d2 =  nn.d2 << 11;

//  q = q - nn
  q.d2 = __sub_cc (q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc_cc(q.d4, nn.d4);
#ifndef DIV_160_96
  q.d5 = __subc   (q.d5, nn.d5);
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  qf= __uint2float_rn(q.d4);
#endif
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf*= 512.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  qi << 23;
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 23 bits
#ifdef CHECKS_MODBASECASE
  nn.d5 =                  nn.d4 >> 9;
#endif  
  nn.d4 = (nn.d4 << 23) + (nn.d3 >> 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
  nn.d1 =  nn.d1 << 23;

// q = q - nn
  q.d1 = __sub_cc (q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
#ifndef CHECKS_MODBASECASE  
  q.d4 = __subc   (q.d4, nn.d4);
#else
  q.d4 = __subc_cc(q.d4, nn.d4);
  q.d5 = __subc   (q.d5, nn.d5);
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  res->d1 = __add_cc(res->d1, qi << 3 );
  res->d2 = __addc  (res->d2, qi >> 29);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

//  q = q - nn
  q.d1 = __sub_cc (q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc   (q.d4, nn.d4);

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf*= 131072.0f;
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  res->d0 = qi << 15;
  res->d1 = __add_cc(res->d1, qi >> 17);
  res->d2 = __addc  (res->d2, 0);
  
// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
#ifndef CHECKS_MODBASECASE
  q.d3 = __subc   (q.d3, nn.d3);
#else
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc   (q.d4, nn.d4);
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d1);
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 = __add_cc (res->d0, qi);
  res->d1 = __addc_cc(res->d1,  0);
  res->d2 = __addc   (res->d2,  0);
  

// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
#ifndef CHECKS_MODBASECASE  
  nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#else
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);
#endif  

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
#ifndef CHECKS_MODBASECASE
  q.d2 = __subc   (q.d2, nn.d2);
#else
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc   (q.d3, nn.d3);
#endif

//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;
  tmp96.d0=q.d0;
  tmp96.d1=q.d1;
  tmp96.d2=q.d2;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
  MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
  MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
  if(cmp_ge_96(tmp96,n))
  {
    res->d0 = __add_cc (res->d0,  1);
    res->d1 = __addc_cc(res->d1,  0);
    res->d2 = __addc   (res->d2,  0);
  }
}
#undef DIV_160_96


#ifndef CHECKS_MODBASECASE
__device__ static void mod_simple_96(int96 *res, int96 q, int96 n, float nf)
#else
__device__ static void mod_simple_96(int96 *res, int96 q, int96 n, float nf, int bit_max64, unsigned int limit, unsigned int *modbasecase_debug)
#endif
/*
res = q mod n
used for refinement in barrett modular multiplication
assumes q < 6n (6n includes "optional mul 2")
*/
{
  float qf;
  unsigned int qi;
  int96 nn;

  qf = __uint2float_rn(q.d2);
  qf = qf * 4294967296.0f + __uint2float_rn(q.d1);
  
  qi=__float2uint_rz(qf*nf);

#ifdef CHECKS_MODBASECASE
/* both barrett based kernels are made for factor candidates above 2^64,
atleast the 79bit variant fails on factor candidates less than 2^64!
Lets ignore those errors...
Factor candidates below 2^64 can occur when TFing from 2^64 to 2^65, the
first candidate in each class can be smaller than 2^64.
This is NOT an issue because those exponents should be TFed to 2^64 with a
kernel which can handle those "small" candidates before starting TF from
2^64 to 2^65. So in worst case we have a false positive which is catched
easily from the primenetserver.
The same applies to factor candidates which are bigger than 2^bit_max for the
barrett92 kernel. If the factor candidate is bigger than 2^bit_max than
usually just the correction factor is bigger than expected. There are tons
of messages that qi is to high (better: higher than expected) e.g. when trial
factoring huge exponents from 2^64 to 2^65 with the barrett92 kernel (during
selftest). The factor candidates might be as high a 2^68 in some of these
cases! This is related to the _HUGE_ blocks that mfaktc processes at once.
To make it short: let's ignore warnings/errors from factor candidates which
are "out of range".
*/
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif

#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  nn.d0 =                          __umul32(n.d0, qi);
  nn.d1 = __umad32hi_cc (n.d0, qi, __umul32(n.d1, qi));
  nn.d2 = __umad32hic   (n.d1, qi, __umul32(n.d2, qi));
#else
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#endif
  
  res->d0 = __sub_cc (q.d0, nn.d0);
  res->d1 = __subc_cc(q.d1, nn.d1);
  res->d2 = __subc   (q.d2, nn.d2);

// perfect refinement not needed, barrett's modular reduction can handle numbers which are a little bit "too big".
/*  if(cmp_ge_96(*res,n))
  {
    sub_96(res, *res, n);
  }*/
}


#if __CUDA_ARCH__ >= 200
  #define KERNEL_MIN_BLOCKS 2
#else
  #define KERNEL_MIN_BLOCKS 1
#endif

__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  int96 exp96,f;
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;
  int bit_max64_32 = 32 - bit_max64; /* used for bit shifting... */

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

  k.d0 = __add_cc (k.d0, __umul32  (k_tab[index], NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_tab[index], NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0 = 1 +                                  __umul32(k.d0, exp96.d0);
  f.d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f.d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f.d1 = __add_cc(f.d1, k.d0);
    f.d2 = __addc  (f.d2, k.d1);  
  }						// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff=__int_as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient
        
  tmp192.d5 = 0x40000000 >> ((bit_max64_32-1) << 1);			// tmp192 = 2^(2*bit_max)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif

  a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);			// a = b / (2^bit_max)
  a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
  a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

  mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)

  a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
  a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

  tmp96.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  tmp96.d2 = __subc   (b.d2, tmp96.d2);

#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  mod_simple_96(&a, tmp96, f, ff, bit_max64, limit, modbasecase_debug);
#endif
  
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_192(&b, a);						// b = a^2

    a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);		// a = b / (2^bit_max)
    a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
    a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

    mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)

    a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
    a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

    tmp96.d0 = __sub_cc (b.d0, tmp96.d0);				// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
    tmp96.d2 = __subc   (b.d2, tmp96.d2);
    
    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    mod_simple_96(&a, tmp96, f, ff, bit_max64, limit, modbasecase_debug);
#endif

    exp<<=1;
  }
  
  if(cmp_ge_96(a,f))				// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
  }

#if defined CHECKS_MODBASECASE && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= 200
  if(cmp_ge_96(a,f) && f.d2)
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f = 1! */  
    index=atomicInc(&RES[0],10000);
    if(index<10)				/* limit to 10 factors per class */
    {
      RES[index*3 + 1]=f.d2;
      RES[index*3 + 2]=f.d1;
      RES[index*3 + 3]=f.d0;
    }
  }
}

__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  int96 exp96,f;
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

  k.d0 = __add_cc (k.d0, __umul32  (k_tab[index], NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_tab[index], NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0 = 1 +                                  __umul32(k.d0, exp96.d0);
  f.d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f.d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f.d1 = __add_cc(f.d1, k.d0);
    f.d2 = __addc  (f.d2, k.d1);  
  }						// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff=__int_as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient

  tmp192.d4 = 0xFFFFFFFF;						// tmp is nearly 2^(80*2)
  tmp192.d3 = 0xFFFFFFFF;
  tmp192.d2 = 0xFFFFFFFF;
  tmp192.d1 = 0xFFFFFFFF;
  tmp192.d0 = 0xFFFFFFFF;

#ifndef CHECKS_MODBASECASE
  div_160_96(&u,tmp192,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(2^(80*2) / f)
#endif

  a.d0 = b.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

  tmp96.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  tmp96.d2 = __subc   (b.d2, tmp96.d2);

#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
  mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

  
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

    tmp96.d0 = __sub_cc (b.d0, tmp96.d0);				// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
    tmp96.d2 = __subc   (b.d2, tmp96.d2);
    
    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
    mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif


//    exp<<=1;
    exp += exp;
  }

  if(cmp_ge_96(a,f))							// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
  }
  
#if defined CHECKS_MODBASECASE && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= 200
  if(cmp_ge_96(a,f) && f.d2)						// factors < 2^64 are not supported by this kernel
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f = 1! */  
    index=atomicInc(&RES[0],10000);
    if(index<10)				/* limit to 10 factors per class */
    {
      RES[index*3 + 1]=f.d2;
      RES[index*3 + 2]=f.d1;
      RES[index*3 + 3]=f.d0;
    }
  }
}

#define TF_BARRETT
#include "tf_common.cu"
  #define TF_BARRETT_79BIT
#include "tf_common.cu"
  #undef TF_BARRETT_79BIT
#undef TF_BARRETT
