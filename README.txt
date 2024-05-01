##################################################
# gr-mfaktc (generalized-repunit-edition) README #
##################################################

Content

0   What is gr-mfaktc (generalized-repunit-edition)?
1   Running gr-mfaktc (generalized-repunit-edition)



#######################
# 0 What is gr-mfaktc #
#######################

gr-mfaktc (generalized-repunit-edition) is a program based on mfaktc (See README-MFAKTC.txt).
It allows factoring generalized repunits with positive as well as negative bases in the range
2 <= |base| < 2^32, doing the work on a CUDA supported graphics card.


###############################
# 1 Running gr-mfaktc         #
###############################

See README.txt for general details.

Example worktodo.txt (for generalized-repunit edition)
-- cut here --
Factor=bla,66362159,64,68
Factor=bla,base=17,1055167,1,64
Factor=bla,base=-97,1055167,1,64
-- cut here --

On Linux run './gr-mfaktc.exe' from the directory with the worktodo.txt file.
On Windows run 'gr-mfaktc-win-64.exe'.

If everything is working as expected this should trial factor
R(10)66362159 [=(10^66362159-1)/9] from 2^64 to 2^68
and after that
R(17)1055167 [=(17^1055167-1)/16] from 2^1 to 2^64, and finally
R(-97)1055167 [=((-97)^1055167-1)/(-98)=(97^1055167+1)/98] from 2^1 to 2^64.
The "bla" string is optional.
The base string defaults to 10 if not given.
