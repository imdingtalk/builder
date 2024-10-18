#!/bin/bash
#SBATCH --partition=Cnode3


cd /public/home/96069/DNA_seq/02DMU/����

# ���г���
if [ -f "yield.DIR" ]; then
  echo "Starting DMUAI with yield.DIR as directive file"
  
  time_now=`date +"%Y-%m-%d %H:%M:%S"`
  echo "start: ${time_now}" > yield.lst
  
  /public/home/96069/software/dmuv6/R5.6-EM64T/bin/dmu1 < yield.DIR >> yield.lst 2>&1
  
  if [ -f MODINF ]; then
    /public/home/96069/software/dmuv6/R5.6-EM64T/bin/dmuai &>> yield.lst
    if [[ $? -eq 0 || -s SOL ]]; then
      printf "program %-27s OK \n" yield >> ../run_yield.log
    else
      echo "yield dmuai failed - Check output files" >> ../run_yield.log
    fi
  else
    echo "yield dmu1 failed - Check output files" >> ../run_yield.log
    echo "DMUAI Not started due to errors in DMU1"
  fi
  
  [ -f SOL ] && mv SOL yield.SOL
  [ -f PAROUT ] && mv PAROUT yield.PAROUT
  [ -f PAROUT_STD ] && mv PAROUT_STD yield.PAROUT_STD
  [ -f LLIK ] && mv LLIK yield.LLIK
  [ -f RESIDUAL ] && mv RESIDUAL yield.RESIDUAL
  
  [ -f CODE_TABLE ] && rm CODE_TABLE
  [ -f DMU1.dir ] && rm DMU1.dir
  [ -f DMUAI.dir ] && rm DMUAI.dir
  [ -f DMU_LOG ] && rm DMU_LOG
  [ -f DUMMY ] && rm DUMMY
  [ -f FSPAKWK ] && rm FSPAKWK
  [ -f Latest_parm ] && rm Latest_parm
  [ -f LEVAL ] && rm LEVAL
  [ -f MODINF ] && rm MODINF
  [ -f PARIN ] && rm PARIN
  [ -f RCDATA_I ] && rm RCDATA_I
  [ -f RCDATA_R ] && rm RCDATA_R
  [ -f AINV* ] && rm AINV*
  [ -f GINV* ] && rm GINV*
  [ -f COR* ] && rm COR*
  [ -f DOM* ] && rm DOM*
  [ -f IBD* ] && rm IBD*
  [ -f PEDFILE* ] && rm PEDFILE*
  rm -f fort.* yield
    
  if [ -s INBREED ]; then
    mv INBREED yield.INBREED
  else
    rm INBREED
  fi
  
  time_now=`date +"%Y-%m-%d %H:%M:%S"`
  echo "end: ${time_now}" >> yield.lst
else
  echo "File yield.DIR not in current directory"
fi
