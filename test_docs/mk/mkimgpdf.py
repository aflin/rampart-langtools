#!/usr/bin/env python3
"""Wrap a baseline JPEG in a minimal image-only PDF (no text layer) -- a
synthetic 'scanned page'.  usage: mkimgpdf.py in.jpg out.pdf [dpi]"""
import struct, sys

def jpeg_info(d):
    i, w, h, ncomp = 2, 0, 0, 3
    while i < len(d):
        if d[i] != 0xFF: i += 1; continue
        m = d[i+1]
        if m in (0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF):
            h, w = struct.unpack('>HH', d[i+5:i+9]); ncomp = d[i+9]; return w,h,ncomp
        if m in (0xD8,0xD9,0x01) or 0xD0 <= m <= 0xD7: i += 2; continue
        i += 2 + struct.unpack('>H', d[i+2:i+4])[0]
    raise SystemExit("no JPEG SOF marker")

src, dst = sys.argv[1], sys.argv[2]
dpi = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
img = open(src,'rb').read()
w,h,ncomp = jpeg_info(img)
pw, ph = w*72.0/dpi, h*72.0/dpi
cs = "/DeviceGray" if ncomp == 1 else ("/DeviceCMYK" if ncomp == 4 else "/DeviceRGB")
content = b"q %.2f 0 0 %.2f 0 0 cm /Im0 Do Q\n" % (pw, ph)

objs = [
 b"<< /Type /Catalog /Pages 2 0 R >>",
 b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
 b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>" % (pw, ph),
 b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (w,h,cs.encode(),len(img)) + img + b"\nendstream",
 b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
]
out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
offs = []
for n, body in enumerate(objs, 1):
    offs.append(len(out))
    out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
xref = len(out)
out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs)+1)
for o in offs: out += b"%010d 00000 n \n" % o
out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs)+1, xref)
open(dst,'wb').write(out)
print("wrote %s  (%dx%d px, %.0fx%.0f pt, %d bytes)" % (dst,w,h,pw,ph,len(out)))
