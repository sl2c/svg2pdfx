#!/usr/bin/env python3

# todo:
# 1) allow all elements to set font-size (and other styles: text-anchor etc.)
# 2) patterns in a separate Object, patternUnits/matrix
# 3) linear/radial gradients -- linear gradients done!
# 4) import external svg
# 5) masking-path-07-b.svg
# 6) BBoxes!!!!! Especially for the root svg -- improved!
# 7) Polyline
# 8) <marker>
# 9) the preserveAspectRatio attribute


'''
svg2pdfx -- convert SVG(s) to PDF efficiently using XObjects
usage: svg2pdfx [-o output.pdf] input1.svg [input2.svg ...]
'''

# globalCount = 0

# These are Page/XObject-s bounding box margins.
# We're talking PDF BBoxes, which have not effect unless they're too small in which case cropping occurs.
# So it's a good idea to keep them on the safe side.
# With this parameter, all BBOxes will be inflated by a factor = 1+margins;
# set margin = 0 for no inflation. A good idea is to keep these margins around 1 to avoid unneeded cropping
# The 'overflow' SVG attribute controls whether cropping should occur at all; we should probably set margins = 1000
# if overflow == 'visible'; it's not implemented yet, though, so we just set margins to a const
margins = 1

# ================================================== Imports

import sys
import string
import re
import math
from math import sin, cos, tan, acos, sqrt, ceil
import base64

from PIL import Image

import io
import zlib
import random
import hashlib
# import codecs

# Try using: github.com/sarnold/pdfrw as it contains many fixes compared to pmaupin's version
from pdfrw import PdfWriter, PdfName, PdfArray, PdfDict, IndirectPdfDict, PdfObject, py23_diffs

from pdfrwx.common import err, warn
from pdfrwx.pdffont import PdfFontUtils

import xml.etree.ElementTree as ET

# ================================================== svgUnits

# These are the SVG units assuming 72 dpi which is appropriate for PDF
# The 'em' unit is relative to the font size: 2em == 2*font_size
# We just assume it's 12 for now, but this needs to be properly coded.
# svgUnits = {'pt':1, 'px':0.75, 'pc':12, 'mm':2.83464, 'cm':28.3464, 'in':72}

svgUnits = {'pt':1, 'px':1, 'pc':12, 'mm':2.83464, 'cm':28.3464, 'in':72, 'em':12}
# svgUnits = {'pt':1.25, 'px':1, 'pc':12, 'mm':2.83464, 'cm':28.3464, 'in':72, 'em':12}

# ================================================== svgIndirectTags

# These are elements that are not to be painted when encountered, only when referenced
# We add 'defs' to it (which cannot be referenced) for ease of processing since it
# is not to be painted by itself either. <title> is excluded since it contains metadata (no drawing)
svgIndirectTags = ['defs', 'altGlyphDef', 'clipPath', 'cursor', 'filter', 'linearGradient',
                    'marker', 'mask', 'pattern', 'radialGradient', 'symbol',
                    'SVGTestCase','title']
        
# ================================================== font config

options = {}
import os
exec_path = os.path.dirname(os.path.abspath(__file__))
ini_path = os.path.join(exec_path,'svg2pdfx.ini')
try:
    with open(ini_path, 'r') as f:
        for line in f:
            l = re.split('=', line.strip(), maxsplit=1)
            if len(l) != 2: print(f'skipping: {line}') ; continue
            options.update({l[0]:l[1]})
    print(f'Read options from {ini_path}:\n{options}')
except:
    print(f'Failed to read options from {ini_path}')

defaultFontDir = options.get('defaultFontDir','fonts')
defaultFontDir = os.path.join(exec_path, defaultFontDir)
defaultUnicodeFont = options.get('defaultUnicodeFont', 'NimbusSans-Regular')

# ================================================== the utilities

utils = PdfFontUtils()

# ================================================== fullProcSet

# This is the full list of PDF processing instruction sets
fullProcSet = [PdfName.PDF, PdfName.Text, PdfName.ImageB, PdfName.ImageC, PdfName.ImageI]

# ================================================== Auxiliary functions

def shift():
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        print(f'Type svg2pdfx -h for usage info')
        sys.exit()

def idGenerator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# ================================================== class attrDict(dict)

class attrDict(dict):
    def __getattr__(self, key): return self.__getitem__(key)
    def __setattr__(self, key, value): self.__setitem__(key,value)       
    def __getitem__(self, key): return dict.__getitem__(self, key) if key in self else None
    def __setitem__(self, key, value):
        if value is not None: dict.__setitem__(self, key, value)
        elif key in self: del self[key]

# ================================================== units()

def units(coordinates: string):
    '''Transform a string containing (comma/space-separated) coordinates or dimensions
       to a float representation in user units (px)
       Returns a list of floats or a single float for a single coordinate.
       ViewPort deminsions arguments are needed to process units given as % of the viewPort'''

    coordList = re.split(r'[\s,]+',coordinates.strip('%')) # Just remove % till proper processing is coded
    for i in range(len(coordList)):
        try:
            c = coordList[i]
            if len(c) > 2 and c[-2:] in svgUnits:
                coordList[i] = float(c[:-2])*svgUnits[c[-2:]]
            else:
                coordList[i] = float(c)
        except:
            warn(f'invalid list of coordinates or dimensions: \'{coordinates}\'; returning None')
            return None
    
    return coordList if len(coordList) > 1 else coordList[0]

# ================================================== rgb_color()

def rgb_color(v:string):
    '''
    Parses SVG color strings and returns color in the form of a triple: [R,G,B]
    '''

    # Full list of SVG colors (https://www.w3.org/TR/SVG11/types.html#ColorKeywords)
    RGB = {'aliceblue':[240,248,255],'antiquewhite':[250,235,215],'aqua':[0,255,255],'aquamarine':[127,255,212],
    'azure':[240,255,255],'beige':[245,245,220],'bisque':[255,228,196],'black':[0,0,0],
    'blanchedalmond':[255,235,205],'blue':[0,0,255],'blueviolet':[138,43,226],'brown':[165,42,42],
    'burlywood':[222,184,135],'cadetblue':[95,158,160],'chartreuse':[127,255,0],'chocolate':[210,105,30],
    'coral':[255,127,80],'cornflowerblue':[100,149,237],'cornsilk':[255,248,220],'crimson':[220,20,60],
    'cyan':[0,255,255],'darkblue':[0,0,139],'darkcyan':[0,139,139],'darkgoldenrod':[184,134,11],
    'darkgray':[169,169,169],'darkgreen':[0,100,0],'darkgrey':[169,169,169],'darkkhaki':[189,183,107],
    'darkmagenta':[139,0,139],'darkolivegreen':[85,107,47],'darkorange':[255,140,0],'darkorchid':[153,50,204],
    'darkred':[139,0,0],'darksalmon':[233,150,122],'darkseagreen':[143,188,143],'darkslateblue':[72,61,139],
    'darkslategray':[47,79,79],'darkslategrey':[47,79,79],'darkturquoise':[0,206,209],'darkviolet':[148,0,211],
    'deeppink':[255,20,147],'deepskyblue':[0,191,255],'dimgray':[105,105,105],'dimgrey':[105,105,105],
    'dodgerblue':[30,144,255],'firebrick':[178,34,34],'floralwhite':[255,250,240],'forestgreen':[34,139,34],
    'fuchsia':[255,0,255],'gainsboro':[220,220,220],'ghostwhite':[248,248,255],'gold':[255,215,0],
    'goldenrod':[218,165,32],'gray':[128,128,128],'grey':[128,128,128],'green':[0,128,0],
    'greenyellow':[173,255,47],'honeydew':[240,255,240],'hotpink':[255,105,180],'indianred':[205,92,92],
    'indigo':[75,0,130],'ivory':[255,255,240],'khaki':[240,230,140],'lavender':[230,230,250],
    'lavenderblush':[255,240,245],'lawngreen':[124,252,0],'lemonchiffon':[255,250,205],'lightblue':[173,216,230],
    'lightcoral':[240,128,128],'lightcyan':[224,255,255],'lightgoldenrodyellow':[250,250,210],'lightgray':[211,211,211],
    'lightgreen':[144,238,144],'lightgrey':[211,211,211],'lightpink':[255,182,193],'lightsalmon':[255,160,122],
    'lightseagreen':[32,178,170],'lightskyblue':[135,206,250],'lightslategray':[119,136,153],'lightslategrey':[119,136,153],
    'lightsteelblue':[176,196,222],'lightyellow':[255,255,224],'lime':[0,255,0],'limegreen':[50,205,50],
    'linen':[250,240,230],'magenta':[255,0,255],'maroon':[128,0,0],'mediumaquamarine':[102,205,170],
    'mediumblue':[0,0,205],'mediumorchid':[186,85,211],'mediumpurple':[147,112,219],'mediumseagreen':[60,179,113],
    'mediumslateblue':[123,104,238],'mediumspringgreen':[0,250,154],'mediumturquoise':[72,209,204],'mediumvioletred':[199,21,133],
    'midnightblue':[25,25,112],'mintcream':[245,255,250],'mistyrose':[255,228,225],'moccasin':[255,228,181],
    'navajowhite':[255,222,173],'navy':[0,0,128],'oldlace':[253,245,230],'olive':[128,128,0],
    'olivedrab':[107,142,35],'orange':[255,165,0],'orangered':[255,69,0],'orchid':[218,112,214],
    'palegoldenrod':[238,232,170],'palegreen':[152,251,152],'paleturquoise':[175,238,238],'palevioletred':[219,112,147],
    'papayawhip':[255,239,213],'peachpuff':[255,218,185],'peru':[205,133,63],'pink':[255,192,203],
    'plum':[221,160,221],'powderblue':[176,224,230],'purple':[128,0,128],'red':[255,0,0],
    'rosybrown':[188,143,143],'royalblue':[65,105,225],'saddlebrown':[139,69,19],'salmon':[250,128,114],
    'sandybrown':[244,164,96],'seagreen':[46,139,87],'seashell':[255,245,238],'sienna':[160,82,45],
    'silver':[192,192,192],'skyblue':[135,206,235],'slateblue':[106,90,205],'slategray':[112,128,144],
    'slategrey':[112,128,144],'snow':[255,250,250],'springgreen':[0,255,127],'steelblue':[70,130,180],
    'tan':[210,180,140],'teal':[0,128,128],'thistle':[216,191,216],'tomato':[255,99,71],
    'turquoise':[64,224,208],'violet':[238,130,238],'wheat':[245,222,179],'white':[255,255,255],
    'whitesmoke':[245,245,245],'yellow':[255,255,0],'yellowgreen':[154,205,50]}

    try:
        if v in RGB: # colors referred to by name
            rgb = RGB[v] 
        elif v[0] == '#':
            v = v.strip('#') # colors in the hex format #AABBCC
            if len(v)==3: v = ''.join([c+c for c in v])
            if len(v)!=6: err(f'invalid color: {v}')
            rgb = [int(h,16) for h in re.findall('..',v)]
        elif v[:4] == 'rgb(': # colors in the 'rgb(r,g,b)' format
            rgbStrLst = [d.strip(' )') for d in re.split(r'[\s,]+',v[4:])]
            rgb = [int(c) if c[-1] != '%' else int(float(c[:-1])*255/100) for c in rgbStrLst ]
        else: err(f'invalid color: {v}')
    except:
        err(f'invalid color: {v}')
    if len(rgb) != 3 or any(d>255 or d<0 for d in rgb): err(f'invalid color: {v}')

    return rgb

# ================================================== class PATH

class PATH(list):

    def __init__(self, svgPath: str):
        '''Create a list representation (list of str/float lists) of the SVG path (<path d="..">)
           by parsing the d-string. Path commands' numerical arguments are converted to floats'''
        # Path commands names together with the numbers of their arguments
        svgPathCommands = {'M':2,'m':2,'Z':0,'z':0,'L':2,'l':2,'H':1,'h':1,'V':1,'v':1,
                        'C':6,'c':6,'S':4,'s':4,'Q':4,'q':4,'T':2,'t':2,'A':7,'a':7}
        # This is needed for converting M/m commands with > 2 args to a string of M+L(n) commands
        cmdConvert = {'M':'L','m':'l'}
        self.bbox = None
        commands = re.split(r'([a-zA-Z][^a-zA-Z]*)', svgPath.strip())
        commands = [cmd.strip() for cmd in commands]
        commands = [cmd for cmd in commands if len(cmd)>0]
        for cmd in commands:
            try:
                cmdName = cmd[0]
                if cmdName not in svgPathCommands: err(f'invalid path command: {cmd}')
                argStr = cmd[1:].strip()
                argStr = re.sub(r'[.]([0-9]+)',r'.\1 ',argStr)
                argStr = re.sub(r'-',' -',argStr).strip()
                args = re.split(r'[\s,]+',argStr) if len(argStr)>0 else []

                # The args now is a list of strings. Special treatment of the elliptic arc commands is needed:
                # their arguments may include flags ('0' & '1') which may not be comma/space-separated
                if cmdName in ['A','a']:
                    i = 0
                    while (i+4 < len(args)):
                        flag = args[i+3]
                        if len(flag)>1: args.insert(i+4,flag[1:]); args[i+3] = flag[0]
                        flag = args[i+4]
                        if len(flag)>1: args.insert(i+5,flag[1:]); args[i+4] = flag[0]
                        i += 7

                # Convert args from list of strings list of floats
                args = [float(arg) for arg in args]

                # Check that there's a whole number of argument chunks
                m = len(args) # actual number of arguments
                n = svgPathCommands[cmdName] # expected number of arguments in a single arguments chunk
                if n==0 and m>0 or n>0 and (m==0 or m % n != 0): err(f'bad number of arguments: {cmd}')
                if n==0: self.append([cmdName]); continue
                # De-concatenate same-command sequences
                for i in range(0,m,n):
                    self.append([cmdName if cmdName not in cmdConvert or i==0 else cmdConvert[cmdName]] + args[i:i+n])
            except:
                err(f'invalid path command: {cmd}')

        self.normalize()

    def __repr__(self):
        '''Implicit conversion to str'''
        return self.toPdfStream()

    def normalize(self):
        '''Converts all commands and coordinates to absolute (upper-case) versions and sets self.bbox'''
        x=0; y=0 # current point
        xmin = None; xmax = None; ymin = None; ymax = None
        xStart = 0; yStart = 0

        for tokens in self:
            cmd = tokens[0]

            if cmd=='Z' or cmd=='z': x=xStart; y=yStart; pass
            elif cmd=='H': x=tokens[1]
            elif cmd=='h': tokens[1]+=x; x=tokens[1]
            elif cmd=='V': y=tokens[1]
            elif cmd=='v': tokens[1]+=y; y=tokens[1]
            else:
                if len(tokens) < 3: err(f'path command has few arguments: {tokens}')
                if cmd in ['m','l','c','s','q','t']:
                    for i in range(1,len(tokens),2): tokens[i] += x; tokens[i+1] += y
                if cmd == 'a':
                    tokens[6] += x; tokens[7] += y # only the last two arguments are relative
                x=tokens[-2]; y=tokens[-1]

            if cmd in ['M','m']: xStart = x; yStart = y

            tokens[0] = tokens[0].upper()

            xmin = min(xmin,x) if xmin != None else x
            xmax = max(xmax,x) if xmax != None else x
            ymin = min(ymin,y) if ymin != None else y
            ymax = max(ymax,y) if ymax != None else y

        b = [xmin, ymin, xmax, ymax]
        if any(a == None for a in b): self.bbox = None
        else: self.bbox = BOX(b)

    def toPdfStream(self):
        '''Convert internal representation (list of str/float lists) to PDF path commands
           All SVG commands except the ones for quadratic Bezier curves have their exact countarparts in PDF.
           Quadratic Bezier curve commands (Q,q,T,t), are expressed (with fidelity) in terms of the cubic ones:
           https://fontforge.org/docs/techref/bezier.html#converting-truetype-to-postscript.
           Conversion of ellipse-drawing commands (A,a) is not implemented and exits with error'''
        r = ''
        u=0.33333
        v=0.66667
        cx=0; cy=0 # current point
        nx=0; ny=0 # new current point after the current command
        dx=0; dy=0 # diff between (cx,cy) and the last control point of the previous Bezier curve
        cmdPrev=''

        p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')

        for tokens in self:
            cmd = tokens[0]
            a = tokens[1:] # Arguments; slicing copies element's values; we don't modify a[] though
            s=''

            # Determine new current point
            if cmd=='Z': pass
            elif cmd=='H': nx=a[0]
            elif cmd=='V': ny=a[0]
            else: nx=a[-2]; ny=a[-1]

            # Set (dx,dy)=0 if command class changes
            if cmd == 'S' and cmdPrev not in ['C','S'] \
                or cmd == 'T' and cmdPrev not in ['Q','T']:
                    dx=0; dy=0

            # Move commands:
            if cmd == 'M': s = f'{p(nx)} {p(ny)} m\n'
            # Path closing commands
            elif cmd == 'Z': s = 'h\n'
            # Line commands
            elif cmd in ['L','H','V']: s = f'{p(nx)} {p(ny)} l\n'
            # Bezier curve commands:
            elif cmd == 'C': s = f'{p(a[0])} {p(a[1])} {p(a[2])} {p(a[3])} {p(nx)} {p(ny)} c\n'
            elif cmd == 'S': s = f'{p(cx+dx)} {p(cy+dy)} {p(a[0])} {p(a[1])} {p(nx)} {p(ny)} c '
            elif cmd == 'Q': s = f'{p(u*cx+v*a[0])} {p(u*cy+v*a[1])} {p(u*nx+v*a[0])} {p(u*ny+v*a[1])} {p(nx)} {p(ny)} c\n'
            elif cmd == 'T': s = f'{p(cx+v*dx)} {p(cy+v*dy)} {p(u*nx+v*(cx+dx))} {p(u*ny+v*(cy+dy))} {p(nx)} {p(ny)} c\n'
            elif cmd == 'A': s = A2C.a2c(cx,cy,a)
            else:
                err(f'path command invalid or its conversion not implemented: {tokens}')

            # Determine (dx,dy)
            if cmd == 'C': dx=nx-a[2]; dy=ny-a[3]
            elif cmd in ['S','Q']: dx=nx-a[0]; dy=ny-a[1]
            elif cmd == 'T': dx=nx-cx-dx; dy=ny-cy-dy
            else: dx=0; dy=0

            cmdPrev = cmd
            cx=nx; cy=ny

            r+=s

        return r

    # def parseBack(self):
    #     r = ''
    #     for commandLine in self:
    #         r += commandLine[0] + ' ' + ' '.join([f'{a:f}' for a in commandLine[1:]]) + ' '
    #     return r.strip()

class A2C:

    '''A utility class for converting elliptic arcs to Bezier curves.
    The code is a (corrected) Python version of the a2c.js from the svgpath package:
    https://github.com/fontello/svgpath
    All credits & thanks go to the original authors: Sergey Batishchev, Vitaly Puzrin & Alex Kocharin'''

    def unit_vector_angle(ux, uy, vx, vy):
        '''Calculate an angle between two unit vectors.
        Since we measure angle between radii of circular arcs,
        we can use simplified math (without length normalization)
        '''
        sign = -1 if ux * vy - uy * vx < 0 else 1
        dot  = ux * vx + uy * vy
        # Add this to work with arbitrary vectors:
        # dot /= Math.sqrt(ux * ux + uy * uy) * Math.sqrt(vx * vx + vy * vy);
        # rounding errors, e.g. -1.0000000000000002 can screw up this
        if (dot >  1.0): dot =  1.0
        if (dot < -1.0): dot = -1.0
        return sign * acos(dot)

    def get_arc_center(x1, y1, x2, y2, fa, fs, rx, ry, sin_phi, cos_phi):
        '''Convert from endpoint to center parameterization,
        see http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
        Returns: cx, cy, theta1, delta_theta
        '''
        # Step 1.
        # Moving an ellipse so origin will be the middlepoint between our two
        # points. After that, rotate it to line up ellipse axes with coordinate axes
        x1p =  cos_phi*(x1-x2)/2 + sin_phi*(y1-y2)/2
        y1p = -sin_phi*(x1-x2)/2 + cos_phi*(y1-y2)/2
        rx_sq  =  rx * rx
        ry_sq  =  ry * ry
        x1p_sq = x1p * x1p
        y1p_sq = y1p * y1p

        # Step 2.
        # Compute coordinates of the centre of this ellipse (cx', cy') in the new coordinate system.
        radicant = (rx_sq * ry_sq) - (rx_sq * y1p_sq) - (ry_sq * x1p_sq)
        if radicant < 0: radicant = 0 # due to rounding errors it might be e.g. -1.38e-17
        radicant /=   (rx_sq * y1p_sq) + (ry_sq * x1p_sq)
        radicant = sqrt(radicant) * (-1 if fa == fs else 1)
        cxp = radicant *  rx/ry * y1p
        cyp = radicant * -ry/rx * x1p

        # Step 3.
        # Transform back to get centre coordinates (cx, cy) in the original coordinate system.
        cx = cos_phi*cxp - sin_phi*cyp + (x1+x2)/2
        cy = sin_phi*cxp + cos_phi*cyp + (y1+y2)/2

        # Step 4.
        # Compute angles (theta1, delta_theta).
        v1x =  (x1p - cxp) / rx
        v1y =  (y1p - cyp) / ry
        v2x = (-x1p - cxp) / rx
        v2y = (-y1p - cyp) / ry
        theta1 = A2C.unit_vector_angle(1, 0, v1x, v1y)
        delta_theta = A2C.unit_vector_angle(v1x, v1y, v2x, v2y)
        if (fs == 0 and delta_theta > 0): delta_theta -= 2 * math.pi
        if (fs == 1 and delta_theta < 0): delta_theta += 2 * math.pi

        return cx, cy, theta1, delta_theta

    def approximate_unit_arc(theta1, delta_theta):
        '''Approximate one unit arc segment with bézier curves,
        see http://math.stackexchange.com/questions/873224.
        Returns: [p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y] - the list of 4 points of the Bezier curve.'''
        alpha = 4/3 * tan(delta_theta/4)
        x1,y1 = cos(theta1),sin(theta1)
        x2,y2 = cos(theta1 + delta_theta),sin(theta1 + delta_theta)
        return [x1, y1, x1 - y1*alpha, y1 + x1*alpha, x2 + y2*alpha, y2 - x2*alpha, x2, y2]

    def a2c(x1:float, y1:float, arguments:list):
        '''Converts an elliptic arc command arguments to one ore more PDF 'c'-commands (Bezier curves).
        (x1,y1) is the starting point of the arc (current point),
        arguments == [rx, ry, phi, fa, fs, x2, y2] (see SVG spec).
        Returns: string with PDF c-commands'''
        p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
        rx, ry, phi, fa, fs, x2, y2 = arguments
        if (x1 == x2 and y1 == y2): return '' # draw nothing if (x1,y1) == (x2,y2), as the spec says
        if (rx == 0 or ry == 0): return f'{p(x2)} {p(y2)} l\n' # return a straight line, as the spec says

        rx,ry = abs(rx),abs(ry)
        sin_phi = sin(phi * math.pi / 180)
        cos_phi = cos(phi * math.pi / 180)        

        x1p =  cos_phi*(x1-x2)/2 + sin_phi*(y1-y2)/2
        y1p = -sin_phi*(x1-x2)/2 + cos_phi*(y1-y2)/2  

        # Compensate out-of-range radii
        scale = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
        if (scale > 1): rx *= sqrt(scale); ry *= sqrt(scale)

        # Get center parameters: cx, cy, theta1, delta_theta
        cx, cy, theta1, delta_theta = A2C.get_arc_center(x1, y1, x2, y2, fa, fs, rx, ry, sin_phi, cos_phi)     

        # Split an arc to multiple segments, so each segment will be less than pi/2 == 90°
        segments = max(ceil(abs(delta_theta) / (math.pi / 2)), 1)
        delta_theta /= segments
        result = ''
        for i in range(segments):
            curve = A2C.approximate_unit_arc(theta1, delta_theta)
            # We have a bezier approximation of a unit arc,
            # now need to transform back to the original ellipse
            for i in range(0,len(curve),2):
                x,y = curve[i:i+2]
                # scale
                x *= rx; y *= ry 
                # rotate
                xp = cos_phi*x - sin_phi*y
                yp = sin_phi*x + cos_phi*y     
                # translate
                curve[i:i+2] = xp + cx, yp + cy
            result += ' '.join(p(c) for c in curve[2:]) + ' c\n'
            theta1 += delta_theta

        return result


    # def parseArcCmd(self, x1:float, y1:float, arguments:list):
    #     '''Implements: https://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes'''
    #     rx,ry,phi,fa,fs,x2,y2 = arguments
    #     p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
    #     if x1==y2 and y1==y2: return ''
    #     if rx==0 or ry==0: return f'{p(x2)} {p(y2)} l\n'
    #     rx = abs(rx); ry = abs(ry)
    #     phi = divmod(phi,360)[1]
    #     fa = 1 if fa != 0 else 0; fs = 1 if fs != 0 else 0


# ================================================== class CTM

class CTM(list):
    '''A CTM matrix'''

    # -------------------------------------------------- __init__()
    def __init__(self, *arguments):
        '''Creates CTM by setting it to a list or by parsing a string containing SVG transformations.
           See: https://www.w3.org/TR/2001/REC-SVG-20010904/coords.html#TransformAttribute'''
        if len(arguments) == 0: self += [1,0,0,1,0,0] ; return
        if len(arguments) > 1: err('more than one argument: {arguments}')
        
        if isinstance(arguments[0],list):
            if len(arguments[0]) == 6: self += arguments[0]; return
            else: err(f'invalid argument: {arguments[0]}')
        elif not isinstance(arguments[0],str):
            err(f'invalid argument type: {arguments[0]}')
        elif arguments[0] == "": self += [1,0,0,1,0,0] ; return

        transformString = arguments[0]

        transforms = transformString.strip(" )").split(")")
        ctm = CTM()

        for transform in transforms:
            tokens = re.split(r'[\s,()]+', transform.strip(' ,\t\r\n'))
            cmd = tokens[0]
            args = [units(tokens[i]) for i in range(1,len(tokens))]
            n = len(args)
            tr = None
            z=math.pi/180

            if cmd == 'matrix' and n==6: tr=args
            elif cmd == 'translate' and n==1: tr=[1,0,0,1,args[0],0]
            elif cmd == 'translate' and n==2: tr=[1,0,0,1,args[0],args[1]]
            elif cmd == 'scale' and n==1: tr=[args[0],0,0,args[0],0,0]
            elif cmd == 'scale' and n==2: tr=[args[0],0,0,args[1],0,0]
            elif cmd == 'rotate' and n==1: a=args[0]; tr=[cos(a*z),sin(a*z),-sin(a*z),cos(a*z),0,0]
            elif cmd == 'rotate' and n==3:
                a=args[0]; cx=args[1]; cy=args[2]
                tr=CTM([1,0,0,1,cx,cy])
                tr=tr.multiply([cos(a*z),sin(a*z),-sin(a*z),cos(a*z),0,0])
                tr=tr.multiply([1,0,0,1,-cx,-cy])
            elif cmd == 'skewX' and n==1: tr = [1,0,tan(args[0]*z),1,0,0]
            elif cmd == 'skewY' and n==1: tr = [1,tan(args[0]*z),0,1,0,0]

            if tr==None: err(f'invalid transform string: {transformString}')

            ctm = ctm.multiply(tr)

        self.clear()
        self += ctm

    # -------------------------------------------------- multiply ()
    def multiply(self, multiplier: 'CTM'):
        '''Returns CTM obtained by multiplying self (CTM) by the multiplier (CTM) from the right.
           See: https://www.w3.org/TR/2001/REC-SVG-20010904/coords.html#TransformMatrixDefined'''
        try:
            ctm = CTM()
            ctm[0] = self[0]*multiplier[0] + self[2]*multiplier[1]
            ctm[2] = self[0]*multiplier[2] + self[2]*multiplier[3]
            ctm[4] = self[0]*multiplier[4] + self[2]*multiplier[5] + self[4]
            ctm[1] = self[1]*multiplier[0] + self[3]*multiplier[1]
            ctm[3] = self[1]*multiplier[2] + self[3]*multiplier[3]
            ctm[5] = self[1]*multiplier[4] + self[3]*multiplier[5] + self[5]
        except:
            err(f'failed to multiply {self} x {multiplier}')
        return ctm

    def inverse(self):
        '''Return inverse ctm'''
        a,b,c,d,e,f = self
        det = a*d-b*c
        if det == 0: err(f'a degenerate ctm matrix has no inverse: {self}')
        return CTM([d/det, -b/det, -c/det, a/det, (c*f-d*e)/det, (b*e-a*f)/det])

    def equal(self, ctm: 'CTM'):
        '''Returns True if self is equal to ctm'''
        return all(self[i]==ctm[i] for i in range(6))

    def toPdfStream(self):
        '''Return a string representation of the transformation
        that can be inserted into a PDF dictionary stream (the 'cm' operator)'''
        p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
        listStr = [p(a) for a in self]
        return ' '.join(listStr)+' cm\n'

    def toPdfStreamRough(self):
        '''Return a string representation of the transformation
        that can be inserted into a PDF dictionary stream (the 'cm' operator);
        this 'rough' version rounds off the dx/dy portion of the cm operator arguments to 1 decimal point accuracy'''
        p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
        q = lambda x: f'{round(x*10)/10:f}'.rstrip('0').rstrip('.')
        listStr = [p(a) for a in self[0:4]] + [q(self[4]), q(self[5])]
        return ' '.join(listStr)+' cm\n'

# ================================================== class VEC

class VEC(list):
    '''A 2D vector'''

    def __init__(self, vec = [0,0]):
        if len(vec) != 2: err(f'invalid vector: {vec}')
        super().__init__(vec)

    def transform(self, ctm: CTM):
        return VEC([self[0]*ctm[0] + self[1]*ctm[2] + ctm[4], self[0]*ctm[1] + self[1]*ctm[3] + ctm[5]])

# ================================================== class DEST

class DEST:
    '''Destinations: refs, idMaps, md5map, tags'''

    def __init__(self, debug):
        self.debug = debug
        self.refs = [] # list of lists of all refIds, accessed by page number --- !!!!! REPLACE WITH A SET !!!!!
        self.idMaps = [] # maps from an id to a node with node.id == id (destination), accessed by page number
        self.md5Map = {} # an all-pages (doc-wide) map from md5 to a node with node.md4 == md5
        self.fonts = {} # an all-pages (doc-wide) map from font names to fonts; used for caching font search results
        self.tags = {} # an all-pages (doc-wide) map of all node.tag's; used for info purposes

    def getNode(self, id: str, page: int):
        '''Return the node (destination) with node.id == id or, if another (identical) node
           with the same node._md5 is found in self.md5Map (i.e., it had been saved earlier),
           then return that other node.'''
        if id == None: err('id is None')
        if page <= 0: err('page numbering starts with 1')
        if id not in self.idMaps[page-1]:
            warn(f'id={id} on page={page} not found in idMaps')
            return None
        node = self.idMaps[page-1][id]
        if node.id != id: err('node.id != id')
        if node._md5 in self.md5Map:
            node_new = self.md5Map[node._md5]
            if node_new._md5 != node._md5: err('node._md5 != md5')
            if node_new != node:
                if self.debug: print(f'mapping {node.id}[{node._page}] --> {node_new.id}[{node_new._page}]')
            node = node_new
        return node

    def gatherReferences(self, node: 'XNODE'):
        '''
        Collects all refIds encountered in the attributes of the node and all of its kids recursively
        and store these refIds in a map: self.refs[page] = refList.
        The self.refs lists are used in self.hashDestinations() to only hash nodes
        that are actually referenced. Note: no checking of the validity of the
        references is done here (see self.validateReferences() for this).
        '''
        p = node._page
        while p > len(self.refs): self.refs.append([])
        for attr in ['clip-path', 'mask', 'href', 'strokePattern', 'fillPattern']:
            if attr in node:
                refId = node[attr]
                if refId not in self.refs[p-1]: self.refs[p-1].append(refId)

        for kid in node._kids: self.gatherReferences(kid)

    def hashDestinations(self, node: 'XNODE'):
        '''Hashes all destinations (nodes with node.id != None) recursively for the node
        and all of its kids. Destinations are hashed in two maps:
        self.idMaps[page][id] and self.md5Map[md5] (the latter is a single map for all pages)
        '''
        p = node._page
        if p <= 0: err('page numbering starts with 1')
        while p > len(self.idMaps): self.idMaps.append({})

        if node.id != None and node.id in self.refs[p-1]: # only hash nodes that are actually referenced
            id = node.id
            if node._md5 != None: err(f'md5 already set: {node}')
            nodeHash = node.toHashableString()
            node._md5 = hashlib.md5(nodeHash.encode("utf-8")).hexdigest() if nodeHash != None else None

            if id in self.idMaps[p-1]:
                node_existing = self.idMaps[p-1][id]
                if node._md5 != node_existing._md5:
                    err(f'a node with this id and a different md5 is already in idMaps[{p}]: {node_existing}')
            else:
                self.idMaps[p-1][id] = node
            if node._md5 != None and node._md5 not in self.md5Map: self.md5Map[node._md5] = node

        for kid in node._kids: self.hashDestinations(kid)

    def validateReferences(self, node: 'XNODE'):
        '''
        Important: the call to this function should be preceded by calls to
        node.parseAttributes() and self.hashDestinations(node).
        Validates references contained in the attributes of the node and all of its kids recursively:
        makes sure that reference destinations exist for all of the refIds encountered,
        otherwise generates an error. In a special case of refIds in patternRef attributes
        (node.strokePattern/fillPattern), if the refDest is not found then the patternRef attribute
        is deleted and node.stroke/fill is set to the node.strokeFallBackColor/fillFallBackColor.
        If the latter is None then an error is generated.
        '''
        paintAttr = {'strokePattern':'stroke','fillPattern':'fill'}
        fbColorAttr = {'strokePattern':'strokeFallBackColor','fillPattern':'fillFallBackColor'}

        for attr in ['clip-path', 'mask', 'href','strokePattern','fillPattern']:
            if attr in node and self.getNode(node[attr],node._page) == None:
                if attr in paintAttr: # the special case of patternRef attributes
                    if node[fbColorAttr[attr]] != None: # if the fallBackColor has been specified
                        node[paintAttr[attr]] = node[fbColorAttr[attr]]
                        del node[attr]; del node[fbColorAttr[attr]]
                    else: err('invalid refDest in: {self.tag}.{attr}={self[attr]} and no fallback color')
                else:
                    warn(f'invalid refDest in {node.tag}.{attr}={node[attr]}')

        for kid in node._kids: self.validateReferences(kid)

    def gatherAttributes(self, node: 'XNODE'):
        '''
        Gathers all tags and attributes from the node and all of its kids and stores them
        in a map: self.tags[tag] = attributesList. Use it to learn of all the tags/attributes contained
        in the node and all of its kids.
        '''
        if node.tag not in self.tags: self.tags[node.tag] = []
        for attr in node:
            if attr[0]!='_' and attr!='tag' and attr not in self.tags[node.tag]:
                self.tags[node.tag].append(attr)
        for kid in node._kids: self.gatherAttributes(kid)


# ================================================== class BOX

class BOX(list):

    def __init__(self, box = [0,0,595,842], convert=False):
        '''Creates a BOX: [xmin, ymin, xmax, ymax].
        Set convert = True for the box argument of the form: [xmin, ymin, width, height]'''
        c = [box[0],box[1],box[0]+box[2],box[1]+box[3]] if convert else box
        super().__init__(c)

    def __repr__(self):
        p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
        return '[' + ', '.join([f'{p(b)}' for b in self]) + ']'

    def round(self):
        return [round(b*1000)/1000 for b in self]

    def dimensions(self):
        '''Returns self in the [xmin, ymin, width, height] format'''
        return [self[0],self[1],self[2]-self[0],self[3]-self[1]]

    def minimax(self,a,b): return (a,b) if a<b else (b,a)

    def equal(self, box: 'BOX'):
        '''Return True if self is equal to box'''
        return all(self[i] == box[i] for i in range(4))

    def embed(self, box: 'BOX'):
        '''Returns a BOX obtained from self by minimal enlargment so that box fully fits in it'''
        r = BOX(self)
        r[0] = min(r[0],box[0])
        r[1] = min(r[1],box[1])
        r[2] = max(r[2],box[2])
        r[3] = max(r[3],box[3])
        return r

    def transformFrom(self, box: 'BOX'):
        '''Returns a ctm such that box.transform(ctm) == self'''
        ctm1 = CTM([1,0,0,1,self[0],self[1]])
        ctm2 = CTM([(self[2]-self[0])/(box[2]-box[0]),0,0,(self[3]-self[1])/(box[3]-box[1]),0,0])
        ctm3 = CTM([1,0,0,1,-box[0],-box[1]])
        return ctm1.multiply(ctm2).multiply(ctm3)

    def transform(self, ctm: CTM):
        '''Return a box which is equal to self transformed by the ctm, i.e. ctm x self'''
        xmin,ymin,xmax,ymax = self
        v1 = VEC([xmin,ymin])
        v2 = VEC([xmin,ymax])
        v3 = VEC([xmax,ymin])
        v4 = VEC([xmax,ymax])

        v1 = v1.transform(ctm)
        v2 = v2.transform(ctm)
        v3 = v3.transform(ctm)
        v4 = v4.transform(ctm)

        xmin = min(min(v1[0],v2[0]),min(v3[0],v4[0]))
        xmax = max(max(v1[0],v2[0]),max(v3[0],v4[0]))
        ymin = min(min(v1[1],v2[1]),min(v3[1],v4[1]))
        ymax = max(max(v1[1],v2[1]),max(v3[1],v4[1]))
        return BOX([xmin,ymin,xmax,ymax])

    def scale(self, scale = 1):
        '''Returns a box which is equal to self scaled with self's center as a fixed point'''
        x = (self[0]+self[2])/2
        y = (self[1]+self[3])/2
        ctm1 = CTM([1,0,0,1,-x,-y])
        ctm2 = CTM([scale,0,0,scale,0,0])
        ctm3 = CTM([1,0,0,1,x,y])
        return self.transform(ctm1).transform(ctm2).transform(ctm3)
        

# ================================================== class STATE

class STATE:
    '''A state is everything that is inherited by kids from parents '''
    contents: PdfDict
    resources: PdfDict
    viewPort: BOX
    paintCmd: str
    paintColor: str
    isClipPath: bool
    fontName: str
    fontSize: float
    debug: bool

    def __init__(self, contents: PdfDict, resources: PdfDict,
                 viewPort: BOX, paintCmd: str, paintColor: str, isClipPath: bool, fontName: str, fontSize: float,
                 debug: bool):
        self.contents = contents # pdf page/dictionary contents (where the .stream is)
        self.resources = resources # pdf page/dictionary resources (where .ExtGState & .XObject are)
        self.viewPort = viewPort # The initial viewport; this is same as the PDF MediaBox
        self.paintCmd = paintCmd # path-painting command
        self.paintColor = paintColor # this is changed by the color="" attributes, e.g. in <g>
        self.isClipPath = isClipPath
        self.fontName = fontName
        self.fontSize = fontSize
        self.debug = debug # debug flag

    def copy(self):
        return STATE(self.contents, self.resources,
                     self.viewPort, self.paintCmd, self.paintColor, self.isClipPath, self.fontName, self.fontSize,
                     self.debug)

# ================================================== class XIMAGE

class XIMAGE:

    def __init__(self,imageStream: str):
        '''Creates a byte-array gzip/deflate-encoded representation of an SVG image ASCII stream'''
        self.width, self.height, self.BitsPerComponent, self.ColorSpace, self.stream, self.filter \
            = self.parseImageStream(imageStream)

    def parseImageStream(self, imageStream):
        if imageStream[:5] == 'data:':
            header, data = re.split(r'base64[\s+,]*', imageStream,1)
            # print(f'data: {data[-10:]}')
            image = Image.open(io.BytesIO(base64.b64decode(data)))
        else:
            image = Image.open(imageStream)
            with open(imageStream,'rb') as f:
                data = f.read()
        
        w,h = image.size
        # image.save("test.jpg")
        mode_to_bpc = {'1':1, 'L':8, 'LA':8, 'P':8, 'RGB':8, 'RGBA':8, 'CMYK':8, 'YCbCr':8, 'I':8, 'F':8}
        mode_to_cs = {'1':'Gray', 'L':'Gray', 'LA':'Gray', 'P':'RGB', 'RGB':'RGB', 'RGBA':'RGB', 'CMYK': 'CMYK'}
        bpc = mode_to_bpc[image.mode]
        if image.mode not in mode_to_cs: err(f'No colorspace for mode: {image.mode}')
        cs = mode_to_cs[image.mode]
        if image.format == 'PNG':
            if image.mode == 'RGBA': image = image.convert('RGB')
            if image.mode == 'LA': image = image.convert('L')
            stream = zlib.compress(image.tobytes())
            filter = 'FlateDecode'
        elif image.format == 'JPEG':
            stream = base64.b64decode(data)
            if stream[-2:] != b'\xff\xd9': stream += b'\xff\xd9'
            # print(f'stream: {stream[-16:]}')
            filter = 'DCTDecode'
        else:
            err(f'unsupported image format: {image.format}')

        return w,h,bpc,cs,stream,filter


    def toStr(self):
        return f'{self.width} x {self.height} {self.ColorSpace} {self.BitsPerComponent}bpc {len(self.stream)}bytes'

# ================================================== class XNODE

class XNODE(attrDict):

    def __init__(self,node: ET.ElementTree, destinations: DEST, page: int):
        '''Puts ET.ElementTree keys (kids) in self._kids and ET.ElementTree.attrib in self,
           making attributes directly accessible.
           This works as long as there are no node attributes like tag=, kids=, xobj= etc.
           Finally, set self._dest to the destinations argument; use this to look up destinations by refId'''
        super().__init__(node.attrib) # node attributes are in self
        self.tag = node.tag # treat tag as an attribute (which kind of makes sense)
        self.text = node.text
        # all the other variables a prefixed with '_'
        self._kids = [XNODE(elem,destinations,page) for elem in node]
        self._xobj = None
        self._md5 = None
        self._state = None
        self._parsed = False
        self._dest = destinations # destinations, see the DEST class
        self._page = page # page number in PDF, starting from 1

    def __repr__(self):
        return self.toString(recurse = False)

    def toString(self, recurse = False, indent = 0):
        '''String representation of the node and all of its kids; only tags, ids & references are printed'''
        r = ' '*indent + f'<{self.tag}'
        for attr in ['id', 'x', 'y', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry',
                        'transform', 'patternTransform', 'href', 'mask', 'fill', 'stroke',
                        'opacity','fill-opacity','stroke-opacity']:
            if self[attr] == None: continue
            if attr == 'href':
                href = self.href[:32] + '..' if self.href != None and len(self.href) > 32 else self.href
                r += f' href="{href}"'
            else: r += f' {attr}="{self[attr]}"'
        r += ' ...>'
        # r += f' BBOX: {self.getBBox()}'
        r += f' port = {self._vpw} x {self._vph}'
        if recurse: r += '\n'.join([kid.toString(recurse, indent+4) for kid in self._kids]) + '\n'
        return r

    def toHashableString(self, indent = 0):
        '''Hashable string representation of the node (with all of its public attributes except .id)
        and all of its kids. Two nodes that have same hash appear identical when drawn, and thus
        subsequent instances of any node with same hash can be substituted by a reference to the first instance.
        '''
        # Some SVG elements use links which, in turn, can contain refIds;
        # now, identical refIds can actually refer to different elements if links are coming from
        # different pages (refIds are page-specific!); this means that just inlcuding refIds as a string
        # in the element's (node's) hash is not enough: we either have to include in the node's hash the hashes
        # of all the refDests for all the refIds in this node's links, or hash no nodes with refIds
        # in the first place (this means they will not be turned into x-objects and will appear in the PDF's
        # content stream as graphics commands). We choose the second option

        # !!! THIS CAN BE IMPROVED: JUST ADD PAGE NUMBER TO THE REFID STRING AND HASH ANYTHING YOU WANT !!!

        if self.href != None or self.mask != None or self['clip-path'] != None: return None
        r = ' '*indent + f'<{self.tag}'
        for attr in self:
            if attr[0] == '_' or attr == 'id': continue # Do not inlcude private attributes or ids in the hash
            r += f' {attr}="{self[attr]}",'
        r += ' ...>\n'
        for kid in self._kids:
            kidHash = kid.toHashableString(indent+4)
            if kidHash == None: return None
            r += kidHash
        return r

    def find(self, value: str):
        '''Returns the first node such that node.tag == value by searhing among self and kids recursively'''
        if self.tag == value: return self
        else:
            for kid in self._kids:
                r = kid.find(value)
                if r != None: return r
            return None

    def parseAttributes(self):
        '''
        Parses node attributes. Coordinates are parsed into floats and the viewBox -- into the BOX class,
        all respecting dimension suffixes. path.d attributes are parsed into PATH classes,
        .transform & .patternTransform -- into CTM classes.
        These non-standard SVG attributes may be set:
        .fillPattern, .fillFallBackColor, .strokePattern, .strokeFallBackColor, image.imageData.
        These attributes may be deleted: .fill, .stroke, image.href
        '''
        if self._parsed: return
        self._parsed = True

        # scaleWidth,scaleHeight  = False,False
        # if self.width != None and self.width[-1] == '%': scaleWidth = True; self.width = self.width.strip('%')
        # if self.height != None and self.height[-1] == '%': scaleHeight = True; self.height = self.height.strip('%')
        
        # Parse lengths
        for attr in ['x','y','width','height','viewBox','cx','cy','r','rx','ry',
                    'x1','y1','x2','y2','points']:
            if attr in self:
                try:
                    self[attr] = units(self[attr])
                except: err(f'failed to parse {self.tag}.{attr}: {self}')

        # Image's href attribute is not really an href: it never contains a refId, so just rename the attribute!
        if self.tag == 'image' and self.href != None: self.imageStream = self.href; del self['href']

        # Parse references in href, mask & clip-path attributes
        # Paint attributes (fill & stroke) can optionally contain refs as well, they are parse at the next step
        for attr in ['href','mask','clip-path']:
            if attr in self:
                refId = self.parseRefId(self[attr])
                if refId == None: err(f'invalid refId {self.tag}.{attr}={self[attr]} in: {self}')
                self[attr] = refId

        # Parses cases of self.fill/stroke == 'url(#patternId) [fallBackColor]'.
        # As a result of parsing, self.fill/stroke is deleted and instead
        # self.fillPattern/strokePattern is set to patternId and
        # self.fillFallBackColor/strokeFallBackColor is set to fallBackColor'''

        # transform self.style='attr:value;..' into the self.attr=value,.. format
        if self.style != None:
            styleList = [s.strip() for s in re.split(r';',self.style) if s != '']
            for s in styleList:
                attr,value=re.split(r':',s,1)
                attr = attr.strip(); value = value.strip()
                if attr == '' or value == '': continue
                self[attr] = value
            del self['style']

        patternAttr = {'fill':'fillPattern','stroke':'strokePattern'}
        fallBackColorAttr = {'fill':'fillFallBackColor','stroke':'strokeFallBackColor'}

        for attr in ['fill','stroke']:
            if attr in self and self[attr][:3] == 'url':
                refList = re.split(r'[\s,]+',self[attr],1)
                patternId = self.parseRefId(refList[0])
                if patternId == None: err(f'invalid patternId in: {self}')
                fallBackColor = refList[1] if len(refList) == 2 and refList[1] != '' else None
                self[patternAttr[attr]] = patternId
                self[fallBackColorAttr[attr]] = fallBackColor
                del self[attr]
        
        # Parse .viewBox, .transform, .patternTransform, .gradientTransform & path.d attributes
        if self.viewBox != None: self.viewBox = BOX(self.viewBox, convert=True)
        if self.transform != None: self.transform = CTM(self.transform)
        if self.patternTransform != None: self.patternTransform = CTM(self.patternTransform)
        if self.gradientTransform != None: self.gradientTransform = CTM(self.gradientTransform)
        if self.d != None: self.d = PATH(self.d)

        # Validate conformance to the SVG specs
        self.validate()
 
        # if scaleWidth:
        #     if self.viewBox != None: fw,fh = self.viewBox[2], self.viewBox[3]
        #     else: fw,fh = 595,842
        #     self.width *= fw/100; self.height *= fh/100
        #     print(f'SCALED DIMS: {self.tag}: {self.width} {self.height}')

        for kid in self._kids:
            kid.parseAttributes()

    def parseRefId(self, refString: str):
        '''Returns refId exctracted from strings like '#refId' or 'url(#refId)' or None if extraction fails'''
        try: return re.split('#', refString.strip(') '),1)[1]
        except: return None

    def validate(self):
        '''Validate element's conformance to the SVG specs'''

        if self.tag == 'line':
            for a in ['x1','y1','x2','y2']: self[a] = self[a] if self[a] != None else 0
        if self.tag == 'rect':
            if self.width == None or self.height == None: err(f'no width/height: {self}')
            if self.width < 0 or self.height < 0: err(f'negative width/height: {self}')
            for a in ['x','y']: self[a] = self[a] if self[a] != None else 0
        if self.tag == 'polygon':
            p = self.points
            if p == None or len(p) == 0 or len(p) % 2 != 0: err(f'invalid points: {self}')
        if self.tag == 'circle':
            if self.r == None or self.r < 0: err(f'missing or negative radius: {self}')
            for a in ['cx','cy']: self[a] = self[a] if self[a] != None else 0
        if self.tag == 'ellipse':
            if self.rx == None or self.rx == None or self.rx < 0 or self.ry < 0:
                err(f'missing or negative radii: {self}')
            for a in ['cx','cy']: self[a] = self[a] if self[a] != None else 0
        if self.tag == 'path':
            if self.d == None: err(f'no path.d: {self}')

    def parseGS(self):
        '''Parse graphics state attributes. Returns GS -- a string containing graphics state
        setting commands and updates self._state.paintCmd -- the path painting command.
        In PDF, most of the graphics state is automatically inherited,
        however path painting commands are not, so we keep them in self._state'''

        # The path painting command is inherited
        cmd = self._state.paintCmd # the original
        CMD = cmd # we'll try to update this

        # Parse the cmd
        paint = {} # This is a map from cmd to the dublets: [stroke,fill]
        cmdToPaint = {'n':[False,False], 'f':[False,True], 'S':[True,False], 'B':[True,True]}
        evenOdd = CMD[-1] == '*' # True if fill-rule/clip-rule is even-odd, False otherwise (for winding-number rule)
        if evenOdd: CMD = CMD[:-1]
        if CMD not in cmdToPaint: err('invalid path painting command: {cmd}')
        paint['stroke'],paint['fill'] = cmdToPaint[CMD]

        # Graphics state string is initialized since it's, essentially, only used to change
        # what's being automatically inherited in PDF
        GS = ''

        # stroke dash parameters are contained in two attributes: stroke-dasharray & stroke-dashoffset
        # we have to remember both during the interation over attributes
        dash = {'array': None, 'phase': '0'}

        opacityStroke = None
        opacityFill = None

        p = lambda x: f'{round(x*100)/100:f}'.rstrip('0').rstrip('.')

        # Process attributes
        for attr in self:
            v = self[attr]
            if v == 'inherit': continue # We inherit by default by passing the state to kids

            # line style-setting attributes

            if attr == 'stroke-width': GS += f'{units(v)} w\n'
            if attr == 'stroke-miterlimit': GS += f'{units(v)} M\n'
            if attr == 'stroke-linecap':
                linecap = {'butt':0, 'round':1, 'square':2}
                if v not in linecap: err(f'invalid attribute value: {attr} = "{v}"')
                GS += f'{linecap[v]} J\n'
            if attr == 'stroke-linejoin':
                linejoin = {'miter':0, 'round':1, 'bevel':2}
                if v not in linejoin: err(f'invalid attribute value: {attr} = "{v}"')
                GS += f'{linejoin[v]} j\n'
            if attr == 'stroke-dasharray':
                dArray = units(v);
                if isinstance(dArray,float): dArray = [dArray]
                dash['array'] = None if v == 'none' else '['+' '.join([str(k) for k in dArray])+']'
            if attr == 'stroke-dashoffset':
                dash['phase'] = f'{units(v)}'
            if attr in ['fill-rule','clip-rule']:
                fillRule = {'evenodd':True,'nonzero':False}
                if v not in fillRule: err(f'invalid fill-rule: {v}')
                evenOdd = fillRule[v]

            # color-setting attributes
            
            if attr == 'color': self._state.paintColor = v

            if attr in ['stroke','fill']:
                if v == 'currentColor': v = self._state.paintColor
                if v == 'none': paint[attr] = False; continue
                paint[attr] = True
                rgb = rgb_color(v)
                colorsStr = ' '.join([f'{p(c/255)}' for c in rgb])
                rg = {'stroke':'RG','fill':'rg'} # the color-setting command
                GS += f'{colorsStr} {rg[attr]}\n'

            # insert references to patterns; the patterns themselves are installed earlier in render()
            # we don't check the validity of the inserted references here; they are checked in 
            if attr == 'strokePattern': GS += f'/Pattern CS\n/{self.strokePattern} SCN\n'
            if attr == 'fillPattern': GS += f'/Pattern cs\n/{self.fillPattern} scn\n'

            # Opacity; do not change opacity inside a clipPath tree:
            # only geometry should be taken into account in constructing the soft mask
            if not self._state.isClipPath:
                if attr == 'stroke-opacity': opacityStroke = float(v)
                if attr == 'fill-opacity': opacityFill = float(v)
                if attr == 'opacity': opacityFill = float(v); opacityStroke = float(v)

        # Dashing
        if all(v != None for v in dash.values()):
            GS += dash['array'] + ' ' + dash['phase'] + ' d\n'

        # Opacity
        if opacityStroke != None or opacityFill != None:
            st = self._state
            if st.resources.ExtGState == None: st.resources.ExtGState = PdfDict()
            egsDict = st.resources.ExtGState
            CA = opacityStroke if opacityStroke != None else None
            ca = opacityFill if opacityFill != None else None

            # See if identical egs is already in st.resources.ExtGState
            egsIdFound = None
            for egsId in egsDict:
                egs = egsDict[egsId]
                if egs.Type == PdfName.ExtGState and egs.BM == PdfName.Normal and egs.CA == CA and egs.ca == ca:
                    egsIdFound = egsId
                    break

            # If not then add a new extended graphics state
            if egsIdFound == None:
                egs = PdfDict(Type = PdfName.ExtGState, BM = PdfName.Normal, CA = CA, ca = ca)
                n = len(st.resources.ExtGState)
                egsIdFound = PdfName(f'gs{n}')
                egsDict[egsIdFound] = egs

            GS += egsIdFound + ' gs\n'

        # Determine the path-painting command
        if paint['stroke'] and paint['fill']: CMD = 'B'
        if paint['stroke'] and not paint['fill']: CMD = 'S'
        if not paint['stroke'] and paint['fill']: CMD = 'f'
        if not paint['stroke'] and not paint['fill']: CMD = 'n' # The default if neither stroke/fill were given

        # # When inside a clipPath tree, filling is all that is done
        # if self._state.isClipPath: CMD = 'f'

        # add a star to the path painting command if the rule is evenodd
        if CMD in ['f','B'] and evenOdd: CMD += '*'

        self._state.paintCmd = CMD

        # Set text font size
        fs = self['font-size']
        if fs != None and fs != 'inherit': self._state.fontSize = units(fs)

        return GS

    def getBBox(self,doNotTransformBBox = False):
        '''Returns BBox = [xmin, ymin, xmax, ymax] by scanning the tree recursively.
           For each node in the tree we either take self.[x, y, y+width, y+height]
           or, if these are unavailable, self.d.bbox (calculated while parsing the SVG d-path), or None.
           Coordinate transforms are also accounted for'''

        box = None
        if self.width != None and self.height != None:
            if self.tag != 'svg' or self.viewBox == None:
                if self.x != None and self.y != None: box = BOX([self.x, self.y, self.width, self.height],convert=True)
                else: box = BOX([0, 0, self.width, self.height])
            else:
                box = self.viewBox

        if self.tag == 'line':
            box = BOX([min(self.x1,self.x2),min(self.y1,self.y2),max(self.x1,self.x2),max(self.y1,self.y2)])

        if self.tag == 'rect':
            box = BOX([self.x,self.y,self.width,self.height],convert=True)

        if self.tag == 'circle':
            box = BOX([self.cx-self.r, self.cy-self.r, self.cx+self.r, self.cy+self.r])

        if self.tag == 'ellipse':
            box = BOX([self.cx-self.rx, self.cy-self.ry, self.cx+self.rx, self.cy+self.ry])

        if self.tag == 'path':
            if box == None: box = self.d.bbox
            else: box = box.embed(self.d.bbox)

        # Avoid cropping lone strictily horizontal/vertial lines
        if box != None:
            x,y,w,h = box.dimensions()
            if w == 0: x -= 1; w = 2; box = BOX([x,y,w,h],convert=True)
            if h == 0: y -= 1; h = 2; box = BOX([x,y,w,h],convert=True)

        if box == None:
            for kid in self._kids:
                if kid not in svgIndirectTags:
                    kidBox = kid.getBBox()
                    if box == None: box = kidBox
                    elif kidBox != None: box = box.embed(kidBox)

        # hrefs are like kids
        if self.tag == 'use' and self.href != None:
            # !!! No need to parse -- already parsed !!!
            # refId = self.parseRefId(self.href)
            refId = self.href
            if refId != None:
                assert re.search(r'#',refId) == None # Make sure it's really parsed
                refDest = self._dest.getNode(refId, self._page)
                refBox = refDest.getBBox()
                if box == None: box = refBox
                elif refBox != None: box = box.embed(refBox)

        x = self.x if self.x != None else 0
        y = self.y if self.y != None else 0
        if box != None and self.transform != None and not doNotTransformBBox:
            box = box.transform(self.transform).transform(CTM([1,0,0,1,x,y]))

        return box

    def createFormXObject(self, doNotTransformBBox = False):
        if self._xobj != None: err('self._xobj is not None')
        bbox = self.getBBox(doNotTransformBBox)
        if bbox == None:
            warn(f'failed to get BBox for {self}, using A4')
            bbox = BOX([0,0,595,842])
        else: bbox = bbox.scale(1+margins)
        self._xobj = IndirectPdfDict(
            Name = PdfName(self.id if self.id != None else 'None'),
            Type = PdfName.XObject,
            Subtype = PdfName.Form,
            FormType = 1,
            BBox = bbox.round(),
            # Resources = PdfDict(ProcSet = fullProcSet)
            Resources = PdfDict()
        )
        self._xobj.stream = ''

    def createImageXObject(self, refId: str):
        '''Returns an image XObject ready to be installed/referenced'''
        image = XIMAGE(self.imageStream)
        if image.width != self.width or image.height != self.height:
            warn(f'read image {image.width} x {image.height}, expected {self.width} x {self.height}')
        xobj = IndirectPdfDict(
            Name = PdfName(refId),
            Type = PdfName.XObject,
            Subtype = PdfName.Image,
            Width = image.width, Height = image.height,
            ColorSpace = PdfName('Device' + image.ColorSpace),
            BitsPerComponent = image.BitsPerComponent,
            Filter = PdfName(image.filter)
            # BBox = BOX([0,0,1,1]).scale(1+margins), --- this entry does not exist for image XObjects!
            # Resources = PdfDict(ProcSet = fullProcSet)
        )
        # Ugly, but necessary: https://github.com/pmaupin/pdfrw/issues/161
        xobj.stream = py23_diffs.convert_load(image.stream)
        return xobj


    def addToXObjectDict(self, xobj: PdfDict, refId = None):
        st = self._state
        if st.resources.XObject == None: st.resources.XObject = PdfDict()
        id = refId if refId != None else 'AutoXObjectID'+ f'{len(st.resources.XObject)}'
        if id not in st.resources.XObject: st.resources.XObject[PdfName(id)] = xobj
        else: warn(f'xobj with id={id} already in the XObject dict')
        return id

    def insertHRef(self, refId: str):
        '''Inserts a reference to the XObject specified by the refId into the contents.stream
            and appends the referenced XObject to the reasources.XObject dict.
            If the referenced XObject does not exist yet it is created and rendered'''
        st = self._state
        refDest = self._dest.getNode(refId, self._page)
        if refDest == None:
            warn(f'refId={refId} on page {self._page} not found, reference not inserted')
            return

        # Append the referenced XObject to the resources.XObject PDFdict
        if refDest._xobj == None:
            refDest.createFormXObject()
            state = st.copy()
            state.contents = refDest._xobj
            state.resources = refDest._xobj.Resources
            refDest.render(state,self._vpw, self._vph)
        self.addToXObjectDict(refDest._xobj, refId)

        # Insert the reference and account for possible presense of refDest.viewBox
        # No need for the 'q/Q' since one is always present inside the referenced XObject; so let's save a little space
        # self.pdfWrite('q\n')
        self.pdfWrite(PdfName(refId) + ' Do\n')
        # self.pdfWrite('Q\n')

    def drawLine(self):
        '''Draws line'''
        self.pdfPaint(f'{self.x1} {self.y1} m {self.x2} {self.y2} l\n')

    def drawRect(self):
        '''Draws rect'''
        if self.width == 0 or self.height == 0: return # as the SVG spec says
        self.pdfPaint(f'{self.x} {self.y} {self.width} {self.height} re\n')

    def drawPolygon(self):
        '''Draw polygon'''
        p = self.points
        path = f'{p[0]} {p[1]} m\n'
        for i in range(2,len(p),2): path += f' {p[i]} {p[i+1]} l\n'
        path += 'h\n'
        self.pdfPaint(path)

    def drawCircle(self):
        '''Draws circle'''
        if self.r == 0: return # as the SVG spec says
        self.drawEllipseWithBezier(self.cx, self.cy, self.r, self.r)

    def drawEllipse(self):
        '''Draws ellipse'''
        if self.rx == 0 or self.ry == 0: return # as the SVG spec says
        self.drawEllipseWithBezier(self.cx, self.cy, self.rx, self.ry)

    def drawEllipseWithBezier(self, cx:float, cy:float, rx:float, ry:float):
        '''Draws ellipse with bezier curvers'''
        k = 0.551784 # 'How to draw an ellipse with Bezier curves': https://www.tinaja.com/glib/ellipse4.pdf
        p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
        self.pdfWrite(f'{p(cx-rx)} {p(cy)} m\n')
        self.pdfWrite(f'{p(cx-rx)} {p(cy+k*ry)} {p(cx-k*rx)} {p(cy+ry)} {p(cx)} {p(cy+ry)} c\n')
        self.pdfWrite(f'{p(cx+k*rx)} {p(cy+ry)} {p(cx+rx)} {p(cy+k*ry)} {p(cx+rx)} {p(cy)} c\n')
        self.pdfWrite(f'{p(cx+rx)} {p(cy-k*ry)} {p(cx+k*rx)} {p(cy-ry)} {p(cx)} {p(cy-ry)} c\n')
        self.pdfWrite(f'{p(cx-k*rx)} {p(cy-ry)} {p(cx-rx)} {p(cy-k*ry)} {p(cx-rx)} {p(cy)} c\n')
        self.pdfWrite(f'h\n')
        self.pdfPaint('')

    # def insertClipPath_BACK(self,refId):
    #     '''Inserts a soft clip based on a soft mask whith the node's alpha as the mask's alha source '''
    #     refDest = self._dest.getNode(refId, self._page)
    #     self.insertSoftMask(refId,isClipPath = True)

    def insertClipPath(self, refId: str):
        '''Clipping based on graphics operators (no XObject is inserted)'''
        refDest = self._dest.getNode(refId, self._page)
        if refDest.tag != 'clipPath': err(f'expected <clipPath>, but got: {refDest}')
        state = self._state.copy()
        state.isClipPath = True
        refDest.render(state, self._vpw, self._vph)
 
    def insertSoftMask(self, refId: str, isClipPath = False):
        '''Inserts soft mask as an ExtGState. maskSource can be either
        PdfName.Luminosity (e.g., use an inverted image as soft mask)
        or PdfName.Alpha (use group's shapes/disregard color as a soft clip),
        see PDF Ref. sec. 7.5.4'''
        st = self._state
        refDest = self._dest.getNode(refId, self._page)

        # switch to the objectBoundingBox coord system
        if 'objectBoundingBox' in [refDest.clipPathUnits,refDest.maskContentUnits]:
            if self.width == None or self.height == None:
                err(f'missing width/height in {self} required for a refDest with objectBoundingBox units')
            ctm = CTM([self.width,0,0,self.height,0,0])
            self.pdfComment('switch to the objectBoundingBox coord system')
            self.pdfWrite(ctm.toPdfStream())

        # Create and render the transparency group associated with the mask node (refDest)
        if refDest._xobj == None:
            refDest.createFormXObject()
            # Booleans are not automatically converted, so we need to wrap them:
            # https://github.com/pmaupin/pdfrw/issues/127
            refDest._xobj.Group = IndirectPdfDict(
                CS = PdfName.DeviceRGB,
                I = PdfObject('true'), # Isolated, i.e. mask's background is transparent
                K = PdfObject('false'), # Knockout is false, see PDF Ref
                S = PdfName.Transparency
            )
            state = st.copy()
            state.contents = refDest._xobj
            state.resources = refDest._xobj.Resources
            state.paintCmd = 'f'
            state.fontSize = 12
            state.fontName = 'Helvetica'
            if isClipPath: state.isClipPath = True
            refDest.render(state, self._vpw, self._vph)

        # Create a soft mask ExtGState (if not yet) and set its SMask.G to the transparency group
        if st.resources.ExtGState == None: st.resources.ExtGState = PdfDict()
        if refId not in st.resources.ExtGState:
            st.resources.ExtGState[PdfName(refId)] = PdfDict(
                Type = PdfName.ExtGState,
                BM = PdfName.Multiply,
                CA = 1, ca = 1,
                SMask = PdfDict(
                    BC = [0,0,0], # Backdrop color (black)
                    S = PdfName.Alpha if isClipPath else PdfName.Luminosity, # source of mask's alpha
                    G = refDest._xobj # Transparency group XObject to be used as mask
                )
            )

        # Insert the mask by setting the graphics state:
        self.pdfWrite(PdfName(refId) + ' gs\n')

        # switch back from the objectBoundingBox coord system
        if 'objectBoundingBox' in [refDest.clipPathUnits,refDest.maskContentUnits]:
            self.pdfComment('switch back from the objectBoundingBox coord system')
            self.pdfWrite(ctm.inverse().toPdfStream())

    def installPattern(self, refId: str):
        '''Insert a pattern, but do not use it yet; it will be referred to (used) later in ParseGS().
           Nested pattern (those that href other patterns) are not supported yet'''
        st = self._state
        refDest = self._dest.getNode(refId, self._page)

        if refDest.tag == 'pattern':
            
            XObj = IndirectPdfDict(
                Name = PdfName(refId),
                Type = PdfName.Pattern,
                stream = ''
            )
            if None in [refDest.width, refDest.height]: err(f'pattern has no width/height: {refDest}')
            # XObj.BBox = refDest.getBBox().scale(1+margins)
            XObj.BBox = PdfArray([0,0,refDest.width,refDest.height])

            XObj.PatternType = 1 # Tiling pattern
            XObj.PaintType = 1 # colored tiling pattern (color is taken from the pattern itself)
            XObj.TilingType = 1 # constant spacing
            XObj.XStep = refDest.width
            XObj.YStep = refDest.height
            XObj.Resources = PdfDict()

            state = st.copy()
            state.contents = XObj
            state.resources = XObj.Resources
            state.viewPort = refDest.getBBox() # This affects the viewportCTM below!
            state.paintCmd = 'f'
            state.fontSize = 12
            state.fontName = 'Helvetica'
            state.isClipPath = False

            # Set the pattern xobject's /Matrix. To properly crop with bbox, you can't use just
            # ctmInner (below), you HAVE to use the pattern xobject's /Matrix!
            ctm = CTM()
            if refDest.patternUnits == 'objectBoundingBox': ctm = CTM([self.width,0,0,self.height,0,0])
            if refDest.patternTransform != None: ctm = ctm.multiply(refDest.patternTransform)
            XObj.Matrix = PdfArray(ctm)

            # SVG spec: "Note that [patternContentUnits] has no effect if attribute ‘viewBox’ is specified."
            if refDest.viewBox == None:
                w,h = 1,1
                if refDest.patternContentUnits == 'objectBoundingBox': w = self.width; h = self.height
                if refDest.patternUnits != 'userSpaceOnUse': w /= self.width; h /= self.height
                ctmInner = CTM([w,0,0,h,0,0])
                XObj.stream += '% ctmInner\n'
                XObj.stream += ctmInner.toPdfStream()

            refDest._xobj = XObj
    
            refDest.render(state, self._vpw, self._vph)

        elif refDest.tag == 'linearGradient':

            p = lambda x: round(x*1000)/1000

            # First, part the SVG part

            stops = [(float(kid['offset']), rgb_color(kid['stop-color'])) for kid in refDest._kids if kid.tag == 'stop']
            stops.sort(key = lambda x: x[0])

            if len(stops) < 2: err('# of stops in linearGradient should be >=2')

            domains = [[stops[i][0],stops[i+1][0]] for i in range(len(stops)-1)]

            intervalFunctions = [
                PdfDict(
                    FunctionType = 2, # Sampled function
                    Domain = PdfArray(domains[i]),
                    C0 = PdfArray(p(float(x)/255) for x in stops[i][1]),
                    C1 = PdfArray(p(float(x)/255) for x in stops[i+1][1]),
                    N = 1
                )
                for i in range(len(domains))
            ]

            fullFunction = IndirectPdfDict(
                FunctionType = 3, # Stitching function
                Domain = PdfArray([stops[0][0],stops[-1][0]]),
                Functions = PdfArray(intervalFunctions),
                Bounds = PdfArray([stops[i][0] for i in range(1,len(stops)-1)]),
                Encode = PdfArray(d for c in domains for d in c)
            )

            # Do the geometry

            coords = [float(refDest[x]) for x in ['x1','y1','x2','y2']]

            # SVG spec: "Note that [patternContentUnits] has no effect if attribute ‘viewBox’ is specified."
            if refDest.gradientUnits == 'objectBoundingBox':
                ctmInner = CTM([1/self.width,0,0,1/self.height,0,0])
                v0,v1 = VEC(coords[0:2]), VEC(coords[2:4])
                coords = v0.transform(ctmInner) + v1.transform(ctmInner)

            # Set the linearGradient/pattern xobject's /Matrix. To properly crop with bbox, you can't use just
            # ctmInner (below), you HAVE to use the linearGradient/pattern xobject's /Matrix!
            ctm = CTM()
            if refDest.gradientUnits == 'objectBoundingBox': ctm = CTM([self.width,0,0,self.height,0,0])
            if refDest.gradientTransform != None:
                ctm = ctm.multiply(refDest.gradientTransform)

            # Then, do the PDF part

            XObj = PdfDict(
                Name = PdfName(refId),
                Type = PdfName.Pattern
            )

            XObj.Matrix = PdfArray(ctm)

            XObj.PatternType = 2 # Shading pattern (gradient)
            XObj.Shading = PdfDict(
                ShadingType = 2, # Axial shading
                ColorSpace = PdfName('DeviceRGB'),
                Coords = PdfArray(coords), # The gradient vector
                Domain = PdfArray([stops[0][0],stops[-1][0]]),
                Function = fullFunction,
                Extend = PdfArray([PdfObject('true'),PdfObject('true')])
            )

            refDest._xobj = XObj

        elif refDest.tag == 'radialGradient':

            err('radialGradient')
            refDest._xobj = IndirectPdfDict(
                Type = PdfName('Pattern'),
                PatternType = 2, # Shading pattern (gradient)
                Shading = PdfDict()
            )

        if st.resources.Pattern == None: st.resources.Pattern = PdfDict()
        st.resources.Pattern[PdfName(refId)] = refDest._xobj

    def insertRect(self):
        st = self._state

        # Initialize
        if st.resources.XObject == None: st.resources.XObject = PdfDict()
        refId = 'svg-rect-' + idGenerator(6) # generate id since SVG <image> doesn't have one
        while refId in st.resources.XObject: refId = 'svg-rect-' + idGenerator(6)

        # Append the self._xobj to the resources.XObject IndirectPDFdict
        self.createImageXObject(refId)
        self.addToXObjectDict(self._xobj, refId)

        self._xobj.stream = 'q\n'

        # After all the geometric transformations the rectangle coincides with the viewport!
        xmin,ymin,xmax,ymax = st.viewPort
        self._xobj.stream += f'{xmin} {ymin} {xmax-xmin} {ymax-ymin} re\n'
        # self.pdfPaint(f'{self.x} {self.y} {self.width} {self.height} re\n')
        self._xobj.stream += f'{st.paintCmd}\n'

        self._xobj.stream += 'Q\n'

        self.pdfWrite(f'/{refId} Do\n')

    def insertImage(self):
        '''Inserts a reference to self._xobj in contents.stream, renders image in self._xobj.stream
            and appends self._xobj to the reasources.XObject dict'''
        # Check self
        if self.tag != 'image': err('expected self.tag == "image": {self}')
        st = self._state
        
        # Initialize
        if st.resources.XObject == None: st.resources.XObject = PdfDict()
        refId = 'svg-image-' + idGenerator(6) # generate id since SVG <image> doesn't have one
        while refId in st.resources.XObject: refId = 'svg-image-' + idGenerator(6)

        # Append the self._xobj to the resources.XObject IndirectPDFdict
        xobj = self.createImageXObject(refId)
        self.addToXObjectDict(xobj, refId)

        # Insert the reference
        # Images are rendered to a 1x1 square which is also (due to the SVG-PDF diff) inverted,
        # and so they need to be inverted back, scaled and translated
        print(f'Inserting image: {xobj.Width} x {xobj.Height}, CS = {xobj.ColorSpace}, BPC = {xobj.BitsPerComponent}')
        self.pdfWrite(f'{self.width} 0 0 {-self.height} {0} {self.height} cm\n')
        self.pdfWrite(PdfName(refId) + ' Do\n')

    def dimensions(self, state: STATE):
        '''Returns self.width,self.height or the appropriate viewPort dimenions if the former are None'''
        vp = state.viewPort
        w = self.width if self.width != None else vp[2]-vp[0]
        h = self.height if self.height != None else vp[3]-vp[1]
        return w,h

    def pdfWrite(self, s: str):
        '''Write string to PDF'''
        if self._state.contents.stream == None: self._state.contents.stream = ''
        self._state.contents.stream += s

    # def pdfPaint_BACK(self, path = ''):
    #     '''Paint path to PDF. Choose appropriate paint command based on self._state'''
    #     self.pdfWrite(path + self._state.paintCmd + '\n')

    def pdfPaint(self, path: str):
        '''Paint path to PDF. Choose appropriate paint command based on self._state'''
        cmd = self._state.paintCmd if not self._state.isClipPath else 'W n'
        if cmd == 'W n' and self['clip-rule'] == 'evenodd': cmd = 'W* n'
        self.pdfWrite(path + cmd + '\n')

    def pdfComment(self, s: str):
        '''When self._state.debug == True, insert a comment in PDF'''
        if self._state.debug: self.pdfWrite('% '+s+'\n')

    def isTextEncodable(self, text:str, encoding:str):
        '''Checks if the all chars in text are from the code page specified by the encoding'''
        try: bytes = text.encode(encoding); return True
        except: return False

    def insertText(self):
        '''Insert self.text by setting up fonts and issuing text commands'''
        # if self.text == None: err('text element has no text')
        if self.text == None: self.text = ''
        st = self._state

        # Replacement of '\n' with '' in the default space handling mode is another SVG oddity
        if self.space == 'preserve' or isinstance(self.x, list) or isinstance(self.y, list):
            text = re.sub(r'[\n\r\t]',' ',self.text)
        else:
            t = re.sub(r'\n\r','',self.text)
            t = re.sub(r'\t',' ',t).strip()
            text = re.sub(r'[ ][ ]*',' ',t)

        # font size (and another text style props) are being set in ._state by parseGS()
        if self.tag == 'text': self.pdfWrite(CTM([1,0,0,-1,0,0]).toPdfStream()) # no flipping for <tspan>

        # determine fontName -- this is what the SVG wants to use
        fontName = self['font-family'] if self['font-family'] != None else st.fontName
        if '+' in fontName: fontPrefix,fontName = re.split(r'\+', fontName,1)
        st.fontName = fontName # remember the requested fontName

        # Create the font by fontName; returned is an instance of PdfFont
        forceCID = not self.isTextEncodable(text, 'cp1252')
        font = utils.loadFont([fontName, defaultUnicodeFont], [defaultFontDir], forceCID)

        self.pdfComment(f'TEXT: {font.encode(text)}')

        # Determine text drawing parameters
        x = self.x if self.x != None else 0
        y = self.y if self.y != None else 0
        fn = font.name # this is what that the font is calling itself; may be different from fontName (apart from '/')
        fs = st.fontSize

        # Install font if necessary
        res = st.resources
        if res.Font == None: res.Font = PdfDict()
        if fn not in res.Font:
            print(f'installing font: {fontName} as {font.name}, isCID = {font.is_cid()}')
            res.Font[fn] = font.font

        # Calculate the text width and the shift necessary for text alignment
        width = font.width(text)
        width /= 1000 # the unit box is 1000x1000 in the font's coord system
        shift = 0
        if self['text-anchor'] == 'middle': shift = -0.5*width*fs
        if self['text-anchor'] == 'end': shift = -width*fs

        # draw text; algo depends on whether the x/y are numbers or lists, see <text> in SVG docs
        convert = font.encode
        if isinstance(x, list) or isinstance(y, list):
            if not isinstance(x, list): x = [x]* len(y)
            if not isinstance(y, list): y = [y]* len(x)
            if len(x) != len(y): err('len(text.x) != len(text.y): text.x: {x}, text.y: {y}')
            for i in range(min(len(x),len(text))):
                self.pdfWrite(f'BT {fn} {fs} Tf {x[i]+shift} {-y[i]} Td {convert(text[i])} Tj ET\n')
        else:
            lines = re.split(r'\n',text)
            lines = [convert(line) for line in lines]
            text = ' Tj T* '.join(lines)
            p = lambda x: f'{round(x*1000)/1000:f}'.rstrip('0').rstrip('.')
            self.pdfWrite(f'BT {fn} {fs} Tf {p(x+shift)} {p(-y)} Td {p(fs*1.2)} TL {text} Tj ET\n')

    def drawBBox(self):
        '''Draw elements' BBoxes in transparent gray/red(for container elements); useful for debugging'''
        bbox = self.getBBox()
        if bbox == None: return
        # bbox = bbox.scale(0.95)
        x,y,w,h = bbox.dimensions()
        st = self._state
        if st.resources.ExtGState == None: st.resources.ExtGState = PdfDict()
        if PdfName.DebugGS not in st.resources.ExtGState:
            st.resources.ExtGState[PdfName.DebugGS] = PdfDict(
                # CA = 0.1,
                ca = 0.1,
                BM = PdfName.Normal
            )

        if st.resources.Font == None: st.resources.Font = PdfDict()
        if PdfName.DebugFont not in st.resources.Font:
            st.resources.Font[PdfName.DebugFont] = \
                PdfDict(
                    Type = PdfName.Font,
                    Subtype = PdfName.Type1,
                    BaseFont = PdfName.Helvetica,
                    Encoding = PdfName.WinAnsiEncoding
                )

        self.pdfWrite('q\n')
        self.pdfWrite(f'/DebugGS gs\n')
        if self._xobj != None:
            self.pdfWrite('1 0 0 rg\n')
        else:
            self.pdfWrite('0.33 0.33 0.33 rg\n')
        self.pdfWrite(f'{x} {y} {w} {h} re\n')
        self.pdfWrite('f\n')
        self.pdfWrite('q\n')
        # self.pdfWrite(CTM([w/120,0,0,h/120,x+w/2,y+h/2]).toPdfStream())
        if self.tag == 'symbol':
            self.pdfWrite(CTM([w/200,0,0,w/200,x,y]).toPdfStream())
            self.pdfWrite(CTM([1,0,0,-1,0,0]).toPdfStream())
            self.pdfWrite(f'BT /DebugFont 12 Tf 0 -12 Td ({self.tag}) Tj ET\n')
        elif self.tag == 'clipPath':
            self.pdfWrite(CTM([w/200,0,0,w/200,x,y+h]).toPdfStream())
            self.pdfWrite(CTM([1,0,0,-1,0,0]).toPdfStream())
            self.pdfWrite(f'BT /DebugFont 12 Tf 0 6 Td ({self.tag}) Tj ET\n')
        else:
            self.pdfWrite(CTM([w/200,0,0,w/200,x+w/2,y]).toPdfStream())
            self.pdfWrite(CTM([1,0,0,-1,0,0]).toPdfStream())
            self.pdfWrite(f'BT /DebugFont 12 Tf 0 -12 Td ({self.tag}) Tj ET\n')
        self.pdfWrite('Q\n')
        self.pdfWrite('Q\n')

    def updateGeometry(self,vpw,vph):
        ''' Updates self._vpw & .vph and set self._ctm -- the viewPort dimensions and transformation
        effected by the geometric attributes (x,y,width,height etc.)
        and self.viewBox (if it exists); vpw & vph arguments are parent's viewPort dimensions'''
        viewPortSetters = ['svg','use','image','symbol','foreignObject','pattern']
        w,h = vpw,vph
        ctm = CTM()
        if self.tag in viewPortSetters:
            if self.width != None: w = self.width
            if self.height != None: h = self.height
            x = self.x if self.x != None else 0
            y = self.y if self.y != None else 0
            ctm  = CTM([1,0,0,1,x,y])
        # adjust viewPort ctm acording to viewBox; viewPort itself stays same
        if self.viewBox != None:
            vbx,vby,vbw,vbh = self.viewBox.dimensions()
            # The treatment of .viewBox is different for the <svg> element
            if self.tag != 'svg':
                if w == None or h == None: err(f'undefined viewPort dimensions while processing viewBox: {self}')
                ctm = ctm.multiply(CTM([w/vbw,0,0,h/vbh,0,0])).multiply(CTM([1,0,0,1,-vbx,-vby]))
            else:
                ctm = ctm.multiply(CTM([1,0,0,1,-vbx,-vby]))

        self._ctm, self._vpw, self._vph = ctm, w, h

    def morphIntoPatternedXObject(self, patternId: str):
        '''
        Fill patterns are not local to the GState: PDF Ref 1.7 sec. 4.6.1:
        "[..] the pattern matrix maps pattern space to the default (initial) coordinate space
        of the page. Changes to the page’s transformation matrix that occur within the page’s
        content stream, such as rotation and scaling, have no effect on the pattern".
        This is contrary to how patterns in SVG are expected to behave, and 
        this might be the reason why most existing svg to pdf converters struggle with patterns.

        To work around this, we make every (!) object that uses a fill pattern into an
        XObject, insert the GState for the fill pattern in the XObject's stream before (!)
        any coord transforms and use the XObject in the current stream by reference.
        This way the pattern will transform along the ctm transforms in the page.
        '''
        if self._xobj != None: return # already morphed
        st = self._state

        # create the XObject
        # do not transform BBox since self.transform has by this point already been parsed in render()
        self.createFormXObject(doNotTransformBBox = True)
        selfId = self.addToXObjectDict(self._xobj)

        # insert the ref to the xobject into the current stream close the q/Q block
        self.pdfComment(f'.self references a pattern: make self into an XObject')
        self.pdfWrite(PdfName(selfId) + ' Do\n')
        self.pdfWrite('Q\n')
        self.pdfComment(f'</{self.tag}>')

        # switch streams by changing state: write to the newly created XObject from now on;
        # this includes parsing kids at the end of render()
        st.contents = self._xobj
        st.resources = self._xobj.Resources

        # Start the stream in the newly created XObject
        self.pdfComment(str(self))
        self.pdfWrite('q\n')

        # Installs the pattern in resources;
        # The actual reference to it is generated in parseGS()
        self.installPattern(patternId)
        # The rest of the stream and the Q will be inserted at the end of this function

    # -------------------------------------------------- render()

    def render(self, state: STATE, vpw = None, vph = None):
        '''
        Renders XNODE (SVG) to PDF recursively. The entire context is in state. vpw, vph are
        parent's viewport dimensions
        '''
        # Do not render invisible elements
        if self.visibility in ['hidden','collapse']: return

        # An SVG oddity: "If the 'mask' ('clip-path') property references a 'mask' ('clip-path')
        # element containing no children, the element referencing it should not be rendered."
        for attr in ['mask','clip-path']:
            if attr in self:
                refDest = self._dest.getNode(self[attr],self._page)
                if len(refDest._kids) == 0: return

        self._state = state.copy() # inherit state
        st = self._state # shorthand notation

        # Draw a semi-transparent BBox around the element; uncomment for hard debugging
        # if st.debug:
        #     self.pdfComment('.drawBBox')
        #     self.drawBBox()

        # Save state (q) & print a comment (for debug mode only)
        # do not use q/Q when clipping since the clip path should be in the same graphics state
        # as the visible (referring) elements that are being clipped
        self.pdfComment(str(self))
        stateSaved = False
        if not st.isClipPath: self.pdfWrite('q\n') ; stateSaved = True

        # Updates self._ctm/vpw/vph
        self.updateGeometry(vpw,vph)

        # Parse self.transform; it has to come before the viewPort transform (see below)
        if self.transform != None:
            self.pdfComment('.transform ctm')
            # self.pdfWrite(self.transform.toPdfStream())
            self.pdfWrite(self.transform.toPdfStreamRough())

        # --- A note on the viewport ctm ---
        # In the use/symbol (or any other link-by-ref case), different callers can call the same object
        # The callers can have different viewPort dimensions, and this will affect refDest._ctm.
        # updateGeometry() at the start of render() above partially takes care of this, however
        # the rendering itself may need to be changed. This is a problem if the self has been is made into
        # an XObject which can be rendered only once.
        # An alternative would be to move some of the transforms out of the refDest into the caller's conents
        # (not implemented yet)

        # Parse the viewPort transform; this is the transformation that accounts for self.x/y & self.viewBox
        if self._ctm != CTM():
            self.pdfComment('.viewPort ctm')
            self.pdfWrite(self._ctm.toPdfStream())

        # If the element references patterns then turn itself (!) into an XObject;
        # see help for self.morphIntoPatternedXObject() for more info
        for attr in ['fillPattern','strokePattern']:
            if attr in self: self.morphIntoPatternedXObject(self[attr])

        # Parse graphics state (mostly style) arguments; path-painting cmd is inherited
        GS = self.parseGS()
        if GS != '':
            self.pdfComment('parseGS')
            self.pdfWrite(GS)

        # Parse <clip-path> & <mask>; order is important
        if self['clip-path'] != None: self.pdfComment('.clip-path'); self.insertClipPath(self['clip-path']) # via an XObject
        if self.mask != None: self.pdfComment('.mask'); self.insertSoftMask(self.mask) # via an XObject

        # Parse <use>, <path>, <text> & <image>
        if self.tag == 'use': self.pdfComment('use.href'); self.insertHRef(self.href) # via an XObject
        if self.tag == 'path': self.pdfComment('path.d'); self.pdfPaint(self.d.toPdfStream())
        if self.tag == 'text': self.pdfComment('text'); self.insertText()
        if self.tag == 'tspan': self.pdfComment('tspan'); self.insertText()
        if self.tag == 'image': self.pdfComment('image.href'); self.insertImage()

        # Parse graphics primitives
        if self.tag == 'line': self.pdfComment('line'); self.drawLine()
        if self.tag == 'rect': self.pdfComment('rect'); self.drawRect()
        if self.tag == 'polygon': self.pdfComment('polygon'); self.drawPolygon()
        if self.tag == 'circle': self.pdfComment('circle'); self.drawCircle()
        if self.tag == 'ellipse': self.pdfComment('ellipse'); self.drawEllipse()
 
        # Recurse; skip ref destinations (nodes with id), they are inserted by refs only (see above)
        for kid in self._kids:
            if kid.tag not in svgIndirectTags and kid.display != 'none':
                kid.render(st, self._vpw, self._vph)

        # Undo the transform when painting the clip path, otherwise it will affect the visible elements

        assert st.isClipPath != stateSaved
 
        if st.isClipPath:
            if self._ctm != CTM():
                self.pdfComment('.inverse viewPort ctm')
                self.pdfWrite(self._ctm.inverse().toPdfStream())
            if self.transform != None:
                self.pdfComment('.inverse transform ctm')
                self.pdfWrite(self.transform.inverse().toPdfStream())

        if not st.isClipPath: self.pdfWrite('Q\n')
        self.pdfComment(f'</{self.tag}>')


# ================================================== class SVGVALIDATE

class SVGVALIDATE:
    def __init__(self, svgPaths: list):
        '''Validate SVG files'''
        print(f'Validating SVG files ...')
        for p, svgPath in enumerate(svgPaths):
            print(f'[{p+1}]', end='\r')
            try: svgRoot = ET.parse(svgPath).getroot()
            except: err(f'Failed to parse: {svgPath}')

# ================================================== class SVG2PDF

class SVG2PDF:

    def __init__(self, pdfPath: str, svgPaths: list, debug = False):
        ''' Converts several SVG files to PDF while encoding identical SVG nodes just once
            by utilizing XObjects. Usage: svg2pdf = SVG2PDF(pdfPath, svgPaths), where svgPaths is a list of paths.'''
        self.debug = debug
        pdf = PdfWriter(pdfPath,compress=False)
        self.destinations = DEST(self.debug) # destinations are nodes with ids
        self.pages = []

        print(f'Parsing and converting SVG files ...')
        for p, svgPath in enumerate(svgPaths):
            page = p + 1
            print('+ '+svgPath)

            # First, do everything that you don't need the PDF for
            svgRoot = ET.parse(svgPath).getroot()
            self.stripNamespaces(svgRoot)
            root = XNODE(svgRoot,self.destinations,page)
            root.parseAttributes()
            self.destinations.gatherReferences(root)
            self.destinations.hashDestinations(root)
            self.destinations.validateReferences(root)
            self.destinations.gatherAttributes(root)
            self.pages.append(root)

            # Then create and render the PDF page and add it to the PDF doc
            pdfPage = self.createPdfPage(root)
            pdf.addPage(pdfPage)

        print(f'Exporting {len(self.pages)} pages to: {pdfPath}')
        if self.debug: print(f'Debugging comments have been inserted in PDF; use a text editor to inspect')
        pdf.write()

    def createPdfPage(self, root: XNODE):
        '''Create a new PDF page by rendering the root node of the SVG tree'''
        box = root.find('svg').getBBox() # This is the outermost <svg> node/element
        if box == None: box = BOX([0,0,595,842])
        p = PdfDict(
            Type = PdfName.Page,
            MediaBox = box.transform(CTM([1,0,0,-1,0,0])), # invert the box
            Contents = IndirectPdfDict(),
            Resources = PdfDict(ProcSet = fullProcSet, Font = PdfDict())
        )
        p.Contents.stream = f'% change coords: SVG->PDF\n'+CTM([1,0,0,-1,0,0]).toPdfStream()
        state = STATE(p.Contents, p.Resources, box, 'f', 'black', False, 'Helvetica', 12, self.debug)
        root.render(state)
        return p

    def stripNamespaces(self, el: ET.ElementTree):
        '''Recursively search this element tree, removing namespaces.
        Taken from: https://stackoverflow.com/questions/32546622/suppress-namespace-in-elementtree'''
        if el.tag.startswith("{"):
            el.tag = el.tag.split('}', 1)[1]  # strip namespace
        for k in list(el.attrib.keys()):
            if k.startswith("{"):
                k2 = k.split('}', 1)[1]
                el.attrib[k2] = el.attrib[k]
                del el.attrib[k]
        for child in el:
            self.stripNamespaces(child)


# ================================================== MAIN

if __name__ == "__main__":

    helpMessage = """\
svg2pdfx -- convert SVG(s) to PDF efficiently using XObjects
usage: svg2pdfx [-debug] [-o output.pdf] input1.svg [input2.svg ...]
if -o is not specified, output is written to input1.svg.pdf
    """

    debug = False
    
    shift()
    pdfPath = None
    while sys.argv[0][0] == '-':
        if sys.argv[0] == '-h': print(helpMessage); sys.exit()
        elif sys.argv[0] == '-o': shift(); pdfPath = sys.argv[0]; shift()
        elif sys.argv[0] == '-debug': shift(); debug = True
        else: err(f'invalid key: {sys.argv[0]}')
    print(f'--> {pdfPath}')
    if pdfPath == None: pdfPath = sys.argv[0] + '.pdf'

    svgValidate = SVGVALIDATE(sys.argv)
    print(f'Validated {len(sys.argv)} SVG files')
    svg2pdf = SVG2PDF(pdfPath, sys.argv, debug)

    if debug or True:
        print('---------------------------------')
        print(f'Tags & attributes encountered:')
        print('---------------------------------')
        for tag, attrs in svg2pdf.destinations.tags.items():
            print(f'{tag} = {attrs}')
        print('---------------------------------')
        size1 = sum(len(m) for m in svg2pdf.destinations.idMaps)
        print(f'Size of idMaps: {size1}')
        # print([len(m) for m in svg2pdf.destinations.idMaps])
        size2 = len(svg2pdf.destinations.md5Map)
        print(f'Size of md5Map: {size2}')
        if size2 != 0: print(f'Compression ratio: {size1/size2:f}')
