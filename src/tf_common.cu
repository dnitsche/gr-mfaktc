/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014  Oliver Weihe (o.weihe@t-online.de)

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


#ifdef SHORTCUT_64BIT
extern "C" __host__ int tf_class_64(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_64
#elif defined (SHORTCUT_75BIT)
extern "C" __host__ int tf_class_75(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_75
#else
extern "C" __host__ int tf_class_95(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_95
#endif
{
  size_t size = mystuff->threads_per_grid * sizeof(int);
  int i, index = 0, stream;
  cudaError_t cuda_ret;
  timeval timer;
  timeval timer2;
  unsigned long long int twait = 0;
  int96 factor,k_base;
  int192 b_preinit;
  int shiftcount, logb, maxlogb, count = 0; // logarithm to base ´base´
  unsigned long long int k_diff;
  char string[50];
  int factorsfound = 0;
  bool is_base_negative = mystuff->base < 0;
  unsigned int abs_base = is_base_negative ? (unsigned int) -mystuff->base : (unsigned int) mystuff->base;

  int h_ktab_index = 0;
  int h_ktab_cpu[CPU_STREAMS_MAX];			// the set of h_ktab[N]s currently ownt by CPU
							// 0 <= N < h_ktab_index: these h_ktab[]s are preprocessed
                                                        // h_ktab_index <= N < mystuff.cpu_streams: these h_ktab[]s are NOT preprocessed
  int h_ktab_inuse[NUM_STREAMS_MAX];			// h_ktab_inuse[N] contains the number of h_ktab[] currently used by stream N
  unsigned long long int k_min_grid[CPU_STREAMS_MAX];	// k_min_grid[N] contains the k_min for h_ktab[h_ktab_cpu[N]], only valid for preprocessed h_ktab[]s

  timer_init(&timer);

  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (mystuff->threads_per_grid + threadsPerBlock - 1) / threadsPerBlock;

  unsigned int delay = 1000;

  for(i=0; i<mystuff->num_streams; i++)h_ktab_inuse[i] = i;
  for(i=0; i<mystuff->cpu_streams; i++)h_ktab_cpu[i] = i + mystuff->num_streams;
  for(i=0; i<mystuff->cpu_streams; i++)k_min_grid[i] = 0;
  h_ktab_index = 0;

  shiftcount=0;
  // shiftcount = (int)log2(exponent), how many bits are there to process
  while((1ULL<<shiftcount) < (unsigned long long int)mystuff->exponent)shiftcount++;
//  printf("\n\nshiftcount = %d\n",shiftcount);
  shiftcount-=1;logb=1;
  maxlogb = shiftcount-3;
  // TODO: find maximum working numbers
#if defined (SHORTCUT_75BIT) || defined (SHORTCUT_64BIT)
  if (maxlogb>16) maxlogb=16; // maximum preprocessing which is possible for 64 bit
#else
  if (maxlogb>20) maxlogb=20; // maximum preprocessing which is possible
#endif
  if (abs_base<=10)
  {
    while(logb<maxlogb || 10*logb<mystuff->bit_min*3)	// how much preprocessing is possible
    {
      shiftcount--;
      logb<<=1; // log(x^2)
      if(mystuff->exponent&(1<<(shiftcount)))logb++; // optional mul with abs_base
    }
  }
//  printf("shiftcount = %d\n",shiftcount);
//  printf("logb = %d\n",logb);
  b_preinit.d5=0;b_preinit.d4=0;b_preinit.d3=0;b_preinit.d2=0;b_preinit.d1=0;b_preinit.d0=1;
// just calculate abs_base^logb
#ifdef SHORTCUT_64BIT
  for(i=0; i<logb; i++) mul64(&b_preinit, b_preinit, abs_base);
#elif defined (SHORTCUT_75BIT)
  for(i=0; i<logb; i++) mul75(&b_preinit, b_preinit, abs_base);
#else
  for(i=0; i<logb; i++) mul96(&b_preinit, b_preinit, abs_base);
#endif

/* set result array to 0 */
  cudaMemsetAsync(mystuff->d_RES, 0, 1*sizeof(unsigned int)); //first int of result array contains the number of factors found

#ifdef DEBUG_GPU_MATH
  cudaMemset(mystuff->d_modbasecase_debug, 0, 32*sizeof(int));
#endif

  timer_init(&timer2);
  while((k_min <= k_max) || (h_ktab_index > 0))
  {
/* preprocessing: calculate a ktab (factor table) */
    if((k_min <= k_max) && (h_ktab_index < mystuff->cpu_streams))	// if we have an empty h_ktab we can preprocess another one
    {
      delay = 1000;
      index = h_ktab_cpu[h_ktab_index];

      if(count > mystuff->num_streams)
      {
        twait+=timer_diff(&timer2);
      }
#ifdef DEBUG_STREAM_SCHEDULE
      printf(" STREAM_SCHEDULE: preprocessing on h_ktab[%d] (count = %d)\n", index, count);
#endif

      sieve_candidates(mystuff->threads_per_grid, mystuff->h_ktab[index], mystuff->sieve_primes);
      k_diff=mystuff->h_ktab[index][mystuff->threads_per_grid-1]+1;
      k_diff*=NUM_CLASSES;				/* NUM_CLASSES because classes are mod NUM_CLASSES */

      k_min_grid[h_ktab_index] = k_min;
      h_ktab_index++;

      count++;
      k_min += (unsigned long long int)k_diff;
      timer_init(&timer2);
    }
    else if(mystuff->allowsleep == 1)
    {
      /* no unused h_ktab for preprocessing.
      This usually means that
      a) all GPU streams are busy
      and
      b) we've preprocessed all available CPU streams
      so let's sleep for some time instead of running a busy loop on cudaStreamQuery() */
      my_usleep(delay);

      delay = delay * 3 / 2;
      if(delay > 500000) delay = 500000;
    }


/* try upload ktab and start the calcualtion of a preprocessed dataset on the device */
    stream = 0;
    while((stream < mystuff->num_streams) && (h_ktab_index > 0))
    {
      if(cudaStreamQuery(mystuff->stream[stream]) == cudaSuccess)
      {
#ifdef DEBUG_STREAM_SCHEDULE
        printf(" STREAM_SCHEDULE: found empty stream: = %d (this releases h_ktab[%d])\n", stream, h_ktab_inuse[stream]);
#endif
        h_ktab_index--;
        i                        = h_ktab_inuse[stream];
        h_ktab_inuse[stream]     = h_ktab_cpu[h_ktab_index];
        h_ktab_cpu[h_ktab_index] = i;

        cudaMemcpyAsync(mystuff->d_ktab[stream], mystuff->h_ktab[h_ktab_inuse[stream]], size, cudaMemcpyHostToDevice, mystuff->stream[stream]);

        k_base.d0 =  k_min_grid[h_ktab_index] & 0xFFFFFFFF;
        k_base.d1 =  k_min_grid[h_ktab_index] >> 32;
        k_base.d2 = 0;

        MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]>>>(mystuff->exponent, abs_base, is_base_negative, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES
#ifdef DEBUG_GPU_MATH
                                                                                    , mystuff->d_modbasecase_debug
#endif
                                                                                    );

#ifdef DEBUG_STREAM_SCHEDULE
        printf(" STREAM_SCHEDULE: started GPU kernel on stream %d using h_ktab[%d]\n\n", stream, h_ktab_inuse[stream]);
#endif
#ifdef DEBUG_GPU_MATH
        cudaDeviceSynchronize(); /* needed to get the output from device printf() */
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
        int j, index_count;
        for(i=0; i < (mystuff->num_streams + mystuff->cpu_streams); i++)
        {
          index_count = 0;
          for(j=0; j<mystuff->num_streams; j++)if(h_ktab_inuse[j] == i)index_count++;
          for(j=0; j<mystuff->cpu_streams; j++)if(h_ktab_cpu[j] == i)index_count++;
          if(index_count != 1)
          {
            printf("DEBUG_STREAM_SCHEDULE_CHECK: ERROR: index %d appeared %d times\n", i, index_count);
            printf("  h_ktab_inuse[] =");
            for(j=0; j<mystuff->num_streams; j++)printf(" %d", h_ktab_inuse[j]);
            printf("\n  h_ktab_cpu[] =");
            for(j=0; j<mystuff->cpu_streams; j++)printf(" %d", h_ktab_cpu[j]);
            printf("\n");
          }
        }
#endif
      }
      stream++;
    }
  }

/* wait to finish the current calculations on the device */
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess)printf("per class final cudaDeviceSynchronize failed!\n");

/* download results from GPU */
  cudaMemcpy(mystuff->h_RES, mystuff->d_RES, 32*sizeof(unsigned int), cudaMemcpyDeviceToHost);

#ifdef DEBUG_GPU_MATH
  cudaMemcpy(mystuff->h_modbasecase_debug, mystuff->d_modbasecase_debug, 32*sizeof(int), cudaMemcpyDeviceToHost);
  for(i=0;i<32;i++)if(mystuff->h_modbasecase_debug[i] != 0)printf("h_modbasecase_debug[%2d] = %u\n", i, mystuff->h_modbasecase_debug[i]);
#endif

  mystuff->stats.grid_count = count;
  mystuff->stats.class_time = timer_diff(&timer)/1000;
/* prevent division by zero if timer resolution is too low */
  if(mystuff->stats.class_time == 0)mystuff->stats.class_time = 1;


  if(count > 2 * mystuff->num_streams)mystuff->stats.cpu_wait = (float)twait / ((float)mystuff->stats.class_time * 10);
  else                                mystuff->stats.cpu_wait = -1.0f;

  print_status_line(mystuff);

  if(mystuff->stats.cpu_wait >= 0.0f)
  {
/* if SievePrimesAdjust is enable lets try to get 2 % < CPU wait < 6% */
    if(mystuff->sieve_primes_adjust == 1 && mystuff->stats.cpu_wait > 6.0f && mystuff->sieve_primes < mystuff->sieve_primes_upper_limit && (mystuff->mode != MODE_SELFTEST_SHORT))
    {
      mystuff->sieve_primes *= 9;
      mystuff->sieve_primes /= 8;
      if(mystuff->sieve_primes > mystuff->sieve_primes_upper_limit) mystuff->sieve_primes = mystuff->sieve_primes_upper_limit;
    }
    if(mystuff->sieve_primes_adjust == 1 && mystuff->stats.cpu_wait < 2.0f  && mystuff->sieve_primes > mystuff->sieve_primes_min && (mystuff->mode != MODE_SELFTEST_SHORT))
    {
      mystuff->sieve_primes *= 7;
      mystuff->sieve_primes /= 8;
      if(mystuff->sieve_primes < mystuff->sieve_primes_min) mystuff->sieve_primes = mystuff->sieve_primes_min;
    }
  }


  factorsfound=mystuff->h_RES[0];
  for(i=0; (i<factorsfound) && (i<10); i++)
  {
    factor.d2=mystuff->h_RES[i*3 + 1];
    factor.d1=mystuff->h_RES[i*3 + 2];
    factor.d0=mystuff->h_RES[i*3 + 3];
    print_dez96(factor,string);
    print_factor(mystuff, i, string);
  }
  if(factorsfound>=10)
  {
    print_factor(mystuff, factorsfound, NULL);
  }

  return factorsfound;
}

#undef MFAKTC_FUNC
