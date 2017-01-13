# This is a module listing all related classes/methods for most frequently used file input/out
# Deal mostly with NWChem-related formats

# imports
import os,sys
import re

# exceptions
class FileIOException(Exception):
   pass

# objects


# methods
def read_fei(file_obj):
   """
   read_fei(file_obj)

   Reads any .fei format files

   input
   ----------
   file_obj - An input file object opened by 'open' command

   output
   ----------
   content - A 'list of dictionaries'
             [{'num_particles': nion_frame1, 'energy': E_frame1, 'frame': [[atom1name, x1, y1, z1], [atom2name, x2, y2, z2], ...]}, ...]
             
             A dictionary contains 3 keys:

             num_particles: number of atoms in that frame
             energy: energy of that frame (a.u.)
             frame: Coordinates of all atoms (including name) in that frame in the format [[atom1, x1, y1, z1], [atom2, x2, y2, z2], ..., [atomN, xN, yN, zN]]
                    x, y, z are all in a.u.

             'content' contains n_frame counts of such dictionary.
   """

   # First step, verify whether the filename is indeed an .fei file.
   # If not, raise an exception and quit

   try:
      filename = os.path.basename(file_obj.name)
      if not re.search(re.compile(r'\w+.fei$'), filename):
         raise FileIOException
   except FileIOException:
      print('FileIOException: The input file is not an .fei file. exiting...')
      sys.exit()
      

   # Before we implement the check, let's veify whether file_obj.name returns a name or full path.
   #print(os.path.basename(file_obj))
   #sys.exit()

   content = []
   dummy_count = []
   counter = 0
   numpart = 0
   print('Please wait while the code is reading your file...')
   for line in file_obj:
      # Reset the counter once it goes beyond one frame's iterations
      if counter >= numpart + 5 or numpart == 0:
         counter = 0
         frame_list = []
         frame_dict = {}
      line_list = line.split()
      if len(line_list) == 1:
         # Read the number of particles
         if counter == 0:
            numpart = eval(line_list[0])
            counter += 1
         # Read the energy
         elif counter == 1:
            energy = eval(line_list[0])
            counter += 1
      else:
         if len(line_list) > 3:
            frame_list.append([line_list[1],eval(line_list[3]),eval(line_list[4]),eval(line_list[5])])
            # if at the last line of each frame, append all related values to the dictionary
            if counter == numpart + 4:
               dummy_count.append('TEST!!')
               #print('Appending frame {0}...'.format(len(dummy_count)))
               frame_dict['num_particles'] = numpart
               frame_dict['energy'] = energy
               frame_dict['frame'] = frame_list
               content.append(frame_dict)
         counter += 1

   print('File reading finished!')
   return content


def read_xyz(file_obj):
   """
   read_xyz(file_obj)

   Read any .xyz format file for any numbers of frames

   Input:
   ----------
   file_obj - An opened file object from an xyz file

   Output:
   ----------
   frames - Returns a list of frames that strictly follows the current format of
            [frame1, frame2, ..., frameN], frame_i = [[atom1, x1, y1, z1], ...],
            or a 3-dimensional list in order to be readily used and compatible with
            the 'write_xyz_from_fei_data' method below
   """
   # First step, verify whether the filename is indeed an .xyz file.
   # If not, raise an exception and quit

   try:
      filename = os.path.basename(file_obj.name)
      if not re.search(re.compile(r'\w+.xyz$'), filename):
         raise FileIOException
   except FileIOException:
      print('FileIOException: The input file is not an .xyz file. exiting...')
      sys.exit()

   # Auxilary variables
   line_count = 0
   frames = []
   numpart = 0

   # Frames reading...
   print('Please wait while the code is reading your .xyz file...')
   for line in file_obj:
      if line_count >= numpart+2 or line_count == 0:
         # If reaching the end of any frame, restart the line_counter
         line_count = 0
         local_frame = []
      line_list = line.split()
      if len(line_list) == 1:
         # Read the number of particles
         if line_count == 0:
            numpart = eval(line_list[0])
            line_count += 1
         else:
            # Simply ignore any other possible contents and count the line
            line_count += 1
      else:
         # There are only 2 possibilities for normal lines: either length 4 or 7
         if len(line_list) == 4:
            # Normal lines contain only coordinates, no velocities
            # Coordinates are in angstroms, velocities are in a.u.
            local_frame.append([line_list[0], eval(line_list[1]), eval(line_list[2]), eval(line_list[3])])
            # If the counter reaches last line of each frame, append local_frame to frames
            if line_count == numpart + 1:
               frames.append(local_frame)

         # If the length is 7
         if len(line_list) == 7:
            local_frame.append([line_list[0], eval(line_list[1]), eval(line_list[2]), eval(line_list[3]), eval(line_list[4]), eval(line_list[5]), eval(line_list[6])])
            if line_count == numpart + 1:
               frames.append(local_frame)

         line_count += 1

   # Done file reading. Exiting...
   print('File reading finished! Exiting...')
   return frames


def write_xyz_from_fei_data(frames):
   """
   write_xyz_from_fei_data(frames)

   Write an .xyz format file from the data of .fei file

   input
   ----------
   frames - Can be one frame or multiple frames. Usually in the format of a list of dict['frames'] data.
            Number of frames is determined from the list's length. This is strictly 2-dimensional list
            and will be checked first.
   *args - An optional argument for path. For now this is not yet implemented for I need to better understand
           python's argument mechanics first. For now, let's just save the file at cwd.

   output
   ----------
   N/A - Simply output a file
   """

   # 'frames' has the format [frame1, frame2, ..., frameN] 
   # frame_i = [[atom1, x1, y1, z1], [atom2, x2, y2, z2], ...]

   # Internal method to return a list's dimension. 'frames' needs to strictly has dimension 3!!
   def get_list_dim(input_list, dim=0):
      
      # Check whether input_list is a list instance
      if isinstance(input_list, list):
         
         if input_list == []:
            return dim
         
         # Recursion
         dim += 1
         dim = get_list_dim(input_list[0], dim)
         return dim
      else:
         if dim == 0:
            return -1
         else: 
            return dim

   # Verify the sanity of 'frames'
   try:
      if get_list_dim(frames) != 3:
         raise FileIOException
   except FileIOException:
      print('FileIOException: Bad format of \'frames\' input. The dimension of \'frames\' is strictly 3 even it contains only one frame.')
      print('Exiting...')
      sys.exit(1)

   # For now, write the file in cwd
   xyz = open('frames.xyz','w')

   for a_frame in frames:
      numpart = len(a_frame)
      
      xyz.write('{0}\n\n'.format(numpart))

      for atom in a_frame:
         # Convert to angstroms. The 'space' in formating string give a leading space for positive number, minus sign for negative number
         xyz.write('{0}\t{1: .6f}\t{2: .6f}\t{3: .6f}\n'.format(atom[0], atom[1]*0.529177, atom[2]*0.529177, atom[3]*0.529177))

      # Finish i-th frame. Write a separate space
      xyz.write('\n')

   xyz.close()
