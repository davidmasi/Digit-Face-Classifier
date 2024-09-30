from . import util, constants

# Filled cells = 0
# Empty cells = 1
def Digit(datum):

  features = util.Counter()

  for x in range(constants.DIGIT_DATUM_WIDTH):
    for y in range(constants.DIGIT_DATUM_HEIGHT):
      features[(x,y)] = datum.getPixel(x, y)
      if features[(x,y)] > 0:
          features[(x,y)] = 1

  return features

def Face(datum):
  
  features = util.Counter()

  for x in range(constants.FACE_DATUM_WIDTH):
    for y in range(constants.FACE_DATUM_HEIGHT):
      features[(x,y)] = datum.getPixel(x, y)
      if features[(x,y)] > 0:
        features[(x,y)] = 1

  return features