/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012  Oliver Weihe (o.weihe@t-online.de)
                                      George Woltman (woltman@alum.mit.edu)

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


__device__ static void shl_96(int96 *a)
/* shiftleft a one bit */
{
  a->d0 = __add_cc (a->d0, a->d0);
  a->d1 = __addc_cc(a->d1, a->d1);
  a->d2 = __addc   (a->d2, a->d2);
}


__device__ static void sub_96(int96 *res, int96 a, int96 b)
/* a must be greater or equal b!
res = a - b */
{
  res->d0 = __sub_cc (a.d0, b.d0);
  res->d1 = __subc_cc(a.d1, b.d1);
  res->d2 = __subc   (a.d2, b.d2);
}


__device__ static void square_96_192(int192 *res, int96 a)
/* res = a^2
assuming that a is < 2^95 (a.d2 < 2^31)! */
{
#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      ".reg .u32 a2;\n\t"

      "mul.lo.u32      %0, %6, %6;\n\t"       /* (a.d0 * a.d0).lo */
      "mul.lo.u32      %1, %6, %7;\n\t"       /* (a.d0 * a.d1).lo */
      "mul.hi.u32      %2, %6, %7;\n\t"       /* (a.d0 * a.d1).hi */

      "add.cc.u32      %1, %1, %1;\n\t"       /* 2 * (a.d0 * a.d1).lo */
      "addc.cc.u32     %2, %2, %2;\n\t"       /* 2 * (a.d0 * a.d1).hi */
      "madc.hi.cc.u32  %3, %7, %7, 0;\n\t"    /* (a.d1 * a.d1).hi */
/* highest possible value for next instruction: mul.lo.u32 (N, N) is 0xFFFFFFF9
this occurs for N = {479772853, 1667710795, 2627256501, 3815194443}
We'll use this knowledge later to avoid some two carry steps to %5 */
      "madc.lo.u32     %4, %8, %8, 0;\n\t"    /* (a.d2 * a.d2).lo */
                                              /* %4 <= 0xFFFFFFFA => no carry to %5 needed! */

      "add.u32         a2, %8, %8;\n\t"       /* a2 = 2 * a.d2 */
                                              /* a is < 2^95 so a.d2 is < 2^31 */

      "mad.hi.cc.u32   %1, %6, %6, %1;\n\t"   /* (a.d0 * a.d0).hi */
      "madc.lo.cc.u32  %2, %7, %7, %2;\n\t"   /* (a.d1 * a.d1).lo */
      "madc.lo.cc.u32  %3, %7, a2, %3;\n\t"   /* 2 * (a.d1 * a.d2).lo */
      "addc.u32        %4, %4,  0;\n\t"       /* %4 <= 0xFFFFFFFB => not carry to %5 needed, see above! */

      "mad.lo.cc.u32   %2, %6, a2, %2;\n\t"   /* 2 * (a.d0 * a.d2).lo */
      "madc.hi.cc.u32  %3, %6, a2, %3;\n\t"   /* 2 * (a.d0 * a.d2).hi */
      "madc.hi.cc.u32  %4, %7, a2, %4;\n\t"   /* 2 * (a.d1 * a.d2).hi */
      "madc.hi.u32     %5, %8, %8, 0;\n\t"    /* (a.d2 * a.d2).hi */
      "}"
      : "=r" (res->d0), "=r" (res->d1), "=r" (res->d2), "=r" (res->d3), "=r" (res->d4), "=r" (res->d5)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2));
#else
  asm("{\n\t"
      ".reg .u32 a2, t1;\n\t"

      "mul.lo.u32      %0, %6, %6;\n\t"       /* (a.d0 * a.d0).lo */
      "mul.lo.u32      %1, %6, %7;\n\t"       /* (a.d0 * a.d1).lo */
      "mul.hi.u32      %2, %6, %7;\n\t"       /* (a.d0 * a.d1).hi */

      "add.cc.u32      %1, %1, %1;\n\t"       /* 2 * (a.d0 * a.d1).lo */
      "addc.cc.u32     %2, %2, %2;\n\t"       /* 2 * (a.d0 * a.d1).hi */
      "mul.hi.u32      t1, %7, %7;\n\t"       /* (a.d1 * a.d1).hi */
      "addc.cc.u32     %3, t1,  0;\n\t"
/* highest possible value for next instruction: mul.lo.u32 (N, N) is 0xFFFFFFF9
this occurs for N = {479772853, 1667710795, 2627256501, 3815194443}
We'll use this knowledge later to avoid some two carry steps to %5 */
      "mul.lo.u32      t1, %8, %8;\n\t"       /* (a.d2 * a.d2).lo */
      "addc.u32        %4, t1,  0;\n\t"       /* %4 <= 0xFFFFFFFA => no carry to %5 needed! */

      "add.u32         a2, %8, %8;\n\t"       /* a2 = 2 * a.d2 */
                                              /* a is < 2^95 so a.d2 is < 2^31 */

      "mul.hi.u32      t1, %6, %6;\n\t"       /* (a.d0 * a.d0).hi */
      "add.cc.u32      %1, %1, t1;\n\t"
      "mul.lo.u32      t1, %7, %7;\n\t"       /* (a.d1 * a.d1).lo */
      "addc.cc.u32     %2, %2, t1;\n\t"
      "mul.lo.u32      t1, %7, a2;\n\t"       /* 2 * (a.d1 * a.d2).lo */
      "addc.cc.u32     %3, %3, t1;\n\t"
      "addc.u32        %4, %4,  0;\n\t"       /* %4 <= 0xFFFFFFFB => not carry to %5 needed, see above! */

      "mul.lo.u32      t1, %6, a2;\n\t"       /* 2 * (a.d0 * a.d2).lo */
      "add.cc.u32      %2, %2, t1;\n\t"
      "mul.hi.u32      t1, %6, a2;\n\t"       /* 2 * (a.d0 * a.d2).hi */
      "addc.cc.u32     %3, %3, t1;\n\t"
      "mul.hi.u32      t1, %7, a2;\n\t"       /* 2 * (a.d1 * a.d2).hi */
      "addc.cc.u32     %4, %4, t1;\n\t"
      "mul.hi.u32      t1, %8, %8;\n\t"       /* (a.d2 * a.d2).hi */
      "addc.u32        %5, t1,  0;\n\t"
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
#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      ".reg .u32 a2;\n\t"

      "mul.lo.u32     %0, %5, %5;\n\t"     /* (a.d0 * a.d0).lo */
      "mul.lo.u32     %1, %5, %6;\n\t"     /* (a.d0 * a.d1).lo */
      "mul.hi.u32     %2, %5, %6;\n\t"     /* (a.d0 * a.d1).hi */

      "add.u32        a2, %7, %7;\n\t"     /* shl(a.d2) */

      "add.cc.u32     %1, %1, %1;\n\t"     /* 2 * (a.d0 * a.d1).lo */
      "addc.cc.u32    %2, %2, %2;\n\t"     /* 2 * (a.d0 * a.d1).hi */
      "madc.hi.u32    %3, %5, a2, 0;\n\t"  /* 2 * (a.d0 * a.d2).hi */
                                           /* %3 (res.d3) has some space left because a2 is < 2^17 */

      "mad.hi.cc.u32  %1, %5, %5, %1;\n\t" /* (a.d0 * a.d0).hi */
      "madc.lo.cc.u32 %2, %6, %6, %2;\n\t" /* (a.d1 * a.d1).lo */
      "madc.hi.cc.u32 %3, %6, %6, %3;\n\t" /* (a.d1 * a.d1).hi */
      "madc.lo.u32    %4, %7, %7, 0;\n\t"  /* (a.d2 * a.d2).lo */

      "mad.lo.cc.u32  %2, %5, a2, %2;\n\t" /* 2 * (a.d0 * a.d2).lo */
      "madc.lo.cc.u32 %3, %6, a2, %3;\n\t" /* 2 * (a.d1 * a.d2).lo */
      "madc.hi.u32    %4, %6, a2, %4;\n\t" /* 2 * (a.d1 * a.d2).hi */
      "}"
      : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3), "=r"(res->d4)
      : "r"(a.d0), "r"(a.d1), "r"(a.d2));
#else
  asm("{\n\t"
      ".reg .u32 a2, t1;\n\t"

      "mul.lo.u32     %0, %5, %5;\n\t"     /* (a.d0 * a.d0).lo */
      "mul.lo.u32     %1, %5, %6;\n\t"     /* (a.d0 * a.d1).lo */
      "mul.hi.u32     %2, %5, %6;\n\t"     /* (a.d0 * a.d1).hi */

      "add.u32        a2, %7, %7;\n\t"     /* shl(a.d2) */

      "add.cc.u32     %1, %1, %1;\n\t"     /* 2 * (a.d0 * a.d1).lo */
      "addc.cc.u32    %2, %2, %2;\n\t"     /* 2 * (a.d0 * a.d1).hi */
      "mul.hi.u32     t1, %5, a2;\n\t"     /* 2 * (a.d0 * a.d2).hi */
      "addc.u32       %3, t1,  0;\n\t"     /* %3 (res.d3) has some space left because a2 is < 2^17 */

      "mul.hi.u32     t1, %5, %5;\n\t"     /* (a.d0 * a.d0).hi */
      "add.cc.u32     %1, %1, t1;\n\t"
      "mul.lo.u32     t1, %6, %6;\n\t"     /* (a.d1 * a.d1).lo */
      "addc.cc.u32    %2, %2, t1;\n\t"
      "mul.hi.u32     t1, %6, %6;\n\t"     /* (a.d1 * a.d1).hi */
      "addc.cc.u32    %3, %3, t1;\n\t"
      "mul.lo.u32     t1, %7, %7;\n\t"     /* (a.d2 * a.d2).lo */
      "addc.u32       %4, t1,  0;\n\t"

      "mul.lo.u32     t1, %5, a2;\n\t"     /* 2 * (a.d0 * a.d2).lo */
      "add.cc.u32     %2, %2, t1;\n\t"
      "mul.lo.u32     t1, %6, a2;\n\t"     /* 2 * (a.d1 * a.d2).lo */
      "addc.cc.u32    %3, %3, t1;\n\t"
      "mul.hi.u32     t1, %6, a2;\n\t"     /* 2 * (a.d1 * a.d2).hi */
      "addc.u32       %4, %4, t1;\n\t"
      "}"
      : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3), "=r"(res->d4)
      : "r"(a.d0), "r"(a.d1), "r"(a.d2));
#endif
}

__device__ static void square_96_128(int192 *res, int96 a)
/* res = a^2
this is a stripped down version of square_96_192, it doesn't compute res.d5 and res.d4
and is a little bit faster.
For correct results a must be less than 2^64 (a.d2 == 0) */
{
#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"

      "mul.lo.u32     %0, %4, %4;\n\t"     /* (a.d0 * a.d0).lo */
      "mul.lo.u32     %1, %4, %5;\n\t"     /* (a.d0 * a.d1).lo */
      "mul.hi.u32     %2, %4, %5;\n\t"     /* (a.d0 * a.d1).hi */

      "add.cc.u32     %1, %1, %1;\n\t"     /* 2 * (a.d0 * a.d1).lo */
      "addc.cc.u32    %2, %2, %2;\n\t"     /* 2 * (a.d0 * a.d1).hi */
      "addc.u32       %3, 0, 0;\n\t"       /* propagate carry */

      "mad.hi.cc.u32  %1, %4, %4, %1;\n\t" /* (a.d0 * a.d0).hi */
      "madc.lo.cc.u32 %2, %5, %5, %2;\n\t" /* (a.d1 * a.d1).lo */
      "madc.hi.u32    %3, %5, %5, %3;\n\t" /* (a.d1 * a.d1).hi */
      "}"
      : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3)
      : "r"(a.d0), "r"(a.d1));
#else
  asm("{\n\t"
      ".reg .u32 t1;\n\t"

      "mul.lo.u32     %0, %5, %5;\n\t"     /* (a.d0 * a.d0).lo */
      "mul.lo.u32     %1, %5, %6;\n\t"     /* (a.d0 * a.d1).lo */
      "mul.hi.u32     %2, %5, %6;\n\t"     /* (a.d0 * a.d1).hi */

      "add.cc.u32     %1, %1, %1;\n\t"     /* 2 * (a.d0 * a.d1).lo */
      "addc.cc.u32    %2, %2, %2;\n\t"     /* 2 * (a.d0 * a.d1).hi */
      "addc.u32       %3, 0,  0;\n\t"      /* propagate carry */

      "mul.hi.u32     t1, %5, %5;\n\t"     /* (a.d0 * a.d0).hi */
      "add.cc.u32     %1, %1, t1;\n\t"
      "mul.lo.u32     t1, %6, %6;\n\t"     /* (a.d1 * a.d1).lo */
      "addc.cc.u32    %2, %2, t1;\n\t"
      "mul.hi.u32     t1, %6, %6;\n\t"     /* (a.d1 * a.d1).hi */
      "addc.cc.u32    %3, %3, t1;\n\t"
      "}"
      : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3), "=r"(res->d4)
      : "r"(a.d0), "r"(a.d1), "r"(a.d2));
#endif
}

#ifdef SHORTCUT_64BIT
extern "C" __host__ void mul64(int192 *res, int192 a, int b)
#elif defined (SHORTCUT_75BIT)
extern "C" __host__ void mul75(int192 *res, int192 a, int b)
#else
extern "C" __host__ void mul96(int192 *res, int192 a, int b)
#endif
{
  unsigned long long int full;
  full = (unsigned long long int)a.d0*(unsigned long long int)b;
  res->d0 = (unsigned int)full;
  res->d1 = (unsigned int)(full>>32);

  full = (unsigned long long int)a.d1*(unsigned long long int)b;
  full += ((full & 0xFFFFFFFF) + (unsigned long long int)res->d1) & 0xFFFFFFFF00000000;
  res->d1 += (unsigned int)full;
  res->d2  = (unsigned int)(full>>32);

  full = (unsigned long long int)a.d2*(unsigned long long int)b;
  full += ((full & 0xFFFFFFFF) + (unsigned long long int)res->d2) & 0xFFFFFFFF00000000;
  res->d2 += (unsigned int)full;
  res->d3  = (unsigned int)(full>>32);

  full = (unsigned long long int)a.d3*(unsigned long long int)b;
  full += ((full & 0xFFFFFFFF) + (unsigned long long int)res->d3) & 0xFFFFFFFF00000000;
  res->d3 += (unsigned int)full;
  res->d4  = (unsigned int)(full>>32);

  full = (unsigned long long int)a.d4*(unsigned long long int)b;
  #if !defined(SHORTCUT_75BIT) && !defined(SHORTCUT_64BIT)
  full += ((full & 0xFFFFFFFF) + (unsigned long long int)res->d4) & 0xFFFFFFFF00000000;
  #endif
  res->d4 += (unsigned int)full;
  #if !defined(SHORTCUT_75BIT) && !defined(SHORTCUT_64BIT)
  res->d5  = (unsigned int)(full>>32);
  res->d5 += a.d5*b;
  #endif
}
