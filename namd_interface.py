##### namd_interface.py #####

import numpy as np
import re


# This is hardcoded for now
head_pattern = re.compile(r'CRYST')
first_atom_pattern = re.compile(r'LIT')
end_pattern = re.compile(r'END')


def write_single_pdb(headline, frame_lines, count, pdb_type):
   if pdb_type == 'coord':
      file_obj = open('../results/snapshot{0}.pdb'.format(count), 'w')
   elif pdb_type == 'vel':
      file_obj = open('../results/snapshot{0}.vel'.format(count), 'w')

   file_obj.write(headline)
   for line in frame_lines:
      file_obj.write(line)

   file_obj.close()


def split_multiframe_pdb(pdb_file_obj, pdb_type):
   numframes = 1
   frame_change_flag = False

   for line in pdb_file_obj:
      if head_pattern.search(line):
         headline = line
         frame_lines = []

      if first_atom_pattern.search(line):
         frame_change_flag = True

      if frame_change_flag:
         frame_lines.append(line)

      if end_pattern.search(line):
         frame_change_flag = False
         write_single_pdb(headline, frame_lines, numframes, pdb_type)
         numframes += 1
         frame_lines = []

   return (numframes - 1)


def write_namd_conf_for_force(index, **kwargs):
   # Catching arguments' exceptions
   # Not good implementation. May change this in the future
   try:
      a = kwargs['cell_size']
      dt = kwargs['timestep']
      switchdist = kwargs['switchdist']
      pairlistdist = kwargs['pairlistdist']
      param_path = kwargs['param']
   except KeyError:
      # Hardcode LiCl parameters here
      a = 19.033
      dt = 0.75
      switchdist = 8.5
      pairlistdist = 12.0
      param_path = 'par_water_ions_spce.prm'

   file_obj = open('../results/snapshot{0}_input.conf'.format(index),'w')
    
   file_obj.write('set inputname\t"snapshot{0}"\n'.format(index))
   file_obj.write('structure\tstructure.psf\n')
   file_obj.write('coordinates\t$inputname.pdb\n')
   file_obj.write('velocities\t$inputname.vel\n\n')
    
   file_obj.write('set outputname\t"snapshot{0}.out"\n'.format(index))
   file_obj.write('outputname\t$outputname\n')
   file_obj.write('binaryoutput\toff\n\n')
    
   file_obj.write('paratypecharmm\ton\n')
   file_obj.write('parameters\t{0}\n'.format(param_path))
   file_obj.write('cutoff\t10.0\n')
   file_obj.write('exclude\tscaled1-4\n')
   file_obj.write('1-4scaling\t1.0\n')
   file_obj.write('switching\ton\n')
   file_obj.write('switchdist\t{0}\n'.format(switchdist))
   file_obj.write('pairlistdist\t{0}\n\n'.format(pairlistdist))
    
   file_obj.write('timestep\t{0}\n'.format(dt))
   file_obj.write('rigidbonds\tall\n\n')
    
   file_obj.write('cellbasisvector1\t{0} 0 0\n'.format(a))
   file_obj.write('cellbasisvector2\t0 {0} 0\n'.format(a))
   file_obj.write('cellbasisvector3\t0 0 {0}\n'.format(a))
   file_obj.write('wrapall\ton\n')
   file_obj.write('PME\tyes\n')
   file_obj.write('PMEGridSpacing\t1.0\n\n')
    
   file_obj.write('run 0\n')
   file_obj.write('output onlyforces $inputname\n')
    
   file_obj.close()
