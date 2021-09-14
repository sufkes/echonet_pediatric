#!/usr/bin/env python

import os
import sys
from readAnnotations import getPeakCurve

in_path = str(sys.argv[1])

peak = getPeakCurve(in_path)

print(peak)
print(peak.shape)
