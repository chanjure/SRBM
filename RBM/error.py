#-*-coding: utf-8 -*-

# ---------------- #
# Error controlles #
# ---------------- #

"""
	Error control module
	~~~~~~~~~
"""

def typeErr(fname,x,right):
  """typeErr
    
  Checks type error.

  Parameters
  ----------
  fname : string
    Name of the process that is checked.
  x : data container
    Data container to be checked its type.
  right : type
    The correct type of the input.

  Returns
  -------
  None
    raise error
  """

  if type(x) != right:
    raise TypeError("Err %s : Input type must be %s"%(fname,str(right)))

