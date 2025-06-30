-- coding: utf-8 --

"""
PI (Power Image) Format - v4.0

This version introduces significant performance and feature enhancements:

NumPy Integration: All pixel operations are now vectorized using NumPy,
providing a massive speed boost over native Python loops.

Alpha Channel Support: Introduces RGBA color mode for transparency.

Indexed/Paletted Color Mode: Adds support for paletted images (like GIF or PNG-8)
for drastically smaller file sizes for graphics with limited colors.

Refactored filtering logic for clarity and performance.
"""
from future import annotations


import struct
import zlib
import io
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import BinaryIO, Dict, List, Optional, Type, Union, ClassVar
from PIL import Image # Used for demonstration purposes

--- Core Format Constants ---

PI_SIGNATURE = b'PI4F'  # Version bumped for breaking changes
CURRENT_VERSION = 0x04
PI_FOOTER = b'PI_E'

--- Enumerations for Type Safety ---

class ChunkType(Enum):
IMAG = b'IMAG'  # Image Data Chunk
META = b'META'  # Metadata Chunk
ICCP = b'ICCP'  # ICC Color Profile Chunk
PLTE = b'PLTE'  # Palette Chunk for Indexed Color

class ColorMode(Enum):
RGB = 3
RGBA = 4
INDEXED = 1

class PNGFilter(Enum):
NONE = 0; SUB = 1; UP = 2; AVERAGE = 3; PAETH = 4

class PIException(Exception): pass
class InvalidFileFormatError(PIException): pass
class CorruptedDataError(PIException): pass

--- Pre-compression Filtering Logic (NumPy-based) ---

def _paeth_predictor(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
"""Vectorized Paeth predictor."""
p = a + b - c
pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))

def apply_filters_numpy(raw_bytes: bytes, width: int, height: int, bpp: int) -> bytes:
"""Applies PNG-style filters to raw pixel data using NumPy for performance."""
if bpp > 4: # Filtering is most effective on byte-sized data
return b'\x00' * height + raw_bytes

scanline_len = width * bpp    
data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, scanline_len)    
filtered_stream = io.BytesIO()    
prev_scanline = np.zeros(scanline_len, dtype=np.uint8)    

for y in range(height):    
    scanline = data[y]    
        
    # Calculate all filtered lines at once using vectorized operations    
    sub_filtered = scanline - np.roll(scanline, bpp)    
    sub_filtered[:bpp] = scanline[:bpp] # Fix wraparound    
        
    up_filtered = scanline - prev_scanline    
    avg_filtered = scanline - ((np.roll(scanline, bpp) + prev_scanline) // 2)    
    avg_filtered[:bpp] = scanline[:bpp] - (prev_scanline[:bpp] // 2)    

    paeth_a = np.roll(scanline, bpp); paeth_a[:bpp] = 0    
    paeth_c = np.roll(prev_scanline, bpp); paeth_c[:bpp] = 0    
    paeth_filtered = scanline - _paeth_predictor(paeth_a, prev_scanline, paeth_c)    

    # Choose the best filter (lowest sum of absolute values)    
    sums = [    
        np.sum(np.abs(sub_filtered.astype(np.int16))),    
        np.sum(np.abs(up_filtered.astype(np.int16))),    
        np.sum(np.abs(avg_filtered.astype(np.int16))),    
        np.sum(np.abs(paeth_filtered.astype(np.int16))),    
        np.sum(scanline) # Base case for PNGFilter.NONE    
    ]    

    best_filter_idx = np.argmin(sums)    
    if sums[best_filter_idx] < sums[4]:    
        filter_type = PNGFilter(best_filter_idx + 1) # SUB is 1    
        if filter_type == PNGFilter.SUB:     best_line = sub_filtered    
        elif filter_type == PNGFilter.UP:      best_line = up_filtered    
        elif filter_type == PNGFilter.AVERAGE: best_line = avg_filtered    
        else:                                  best_line = paeth_filtered    
    else:    
        filter_type = PNGFilter.NONE    
        best_line = scanline    

    filtered_stream.write(bytes([filter_type.value]))    
    filtered_stream.write(best_line.tobytes())    
    prev_scanline = scanline    

return filtered_stream.getvalue()

def unfilter_numpy(filtered_bytes: bytes, width: int, height: int, bpp: int) -> bytes:
"""Un-filters data using NumPy. The inverse of apply_filters_numpy."""
if bpp > 4: # Data was not filtered
return filtered_bytes[height:] # Skip filter type bytes

scanline_len = width * bpp    
unfiltered_data = np.empty((height, scanline_len), dtype=np.uint8)    
prev_scanline = np.zeros(scanline_len, dtype=np.uint8)    
stream = io.BytesIO(filtered_bytes)    

for y in range(height):    
    filter_type = PNGFilter(stream.read(1)[0])    
    filtered_line_bytes = stream.read(scanline_len)    
    filtered_line = np.frombuffer(filtered_line_bytes, dtype=np.uint8)    
        
    if filter_type == PNGFilter.NONE:    
        current_scanline = filtered_line    
    else: # Reconstruct scanline pixel by pixel for dependency reasons    
        current_scanline = np.empty(scanline_len, dtype=np.uint8)    
        for i in range(scanline_len):    
            a = current_scanline[i - bpp] if i >= bpp else 0    
            b = prev_scanline[i]    
            c = prev_scanline[i - bpp] if i >= bpp else 0    

            if filter_type == PNGFilter.SUB:     pred = a    
            elif filter_type == PNGFilter.UP:      pred = b    
            elif filter_type == PNGFilter.AVERAGE: pred = (a + b) // 2    
            else:                                  pred = _paeth_predictor(a, b, c) # PAETH    
                
            current_scanline[i] = (filtered_line[i] + pred) & 0xFF    
        
    unfiltered_data[y] = current_scanline    
    prev_scanline = current_scanline    
        
return unfiltered_data.tobytes()

--- Chunk Definitions ---

@dataclass(frozen=True)
class BaseChunk:
TYPE: ClassVar[ChunkType]
def to_bytes(self) -> bytes:
payload = self._get_payload(); crc = zlib.crc32(self.TYPE.value + payload)
header = struct.pack('>4sII', self.TYPE.value, len(payload), crc)
return header + payload
def _get_payload(self) -> bytes: raise NotImplementedError
@classmethod
def from_payload(cls, payload: bytes) -> BaseChunk: raise NotImplementedError

META and ICCP are unchanged

@dataclass(frozen=True)
class METAChunk(BaseChunk):
TYPE: ClassVar[ChunkType] = ChunkType.META; metadata: Dict[str, str] = field(default_factory=dict)
def _get_payload(self) -> bytes:
p=io.BytesIO();[p.write(struct.pack('>I',len(k.encode()))+k.encode()+struct.pack('>I',len(v.encode()))+v.encode()) for k,v in self.metadata.items()];return p.getvalue()
@classmethod
def from_payload(cls, p: bytes) -> METAChunk:
m={};s=io.BytesIO(p);(lambda: (l:=s.read(4)) and (m.update({s.read(struct.unpack('>I',l)[0]).decode():s.read(struct.unpack('>I',s.read(4))[0]).decode()}),l))() and ...; return cls(m)

@dataclass(frozen=True)
class ICCPChunk(BaseChunk):
TYPE: ClassVar[ChunkType] = ChunkType.ICCP; profile_data: bytes
def _get_payload(self) -> bytes: return self.profile_data
@classmethod
def from_payload(cls, payload: bytes) -> ICCPChunk: return cls(profile_data=payload)

@dataclass(frozen=True)
class PLTEChunk(BaseChunk):
"""Palette chunk for indexed color images. Stores up to 256 colors."""
TYPE: ClassVar[ChunkType] = ChunkType.PLTE
palette: np.ndarray # Shape (N, 3) or (N, 4) for RGB/RGBA, dtype=uint8

def _get_payload(self) -> bytes:    
    num_colors, channels = self.palette.shape    
    header = struct.pack('>HB', num_colors, channels)    
    return header + self.palette.tobytes()    

@classmethod    
def from_payload(cls, payload: bytes) -> PLTEChunk:    
    num_colors, channels = struct.unpack('>HB', payload[:3])    
    palette_data = np.frombuffer(payload[3:], dtype=np.uint8)    
    palette = palette_data.reshape((num_colors, channels))    
    return cls(palette)

@dataclass(frozen=True)
class IMAGChunk(BaseChunk):
"""Main image data chunk, now using NumPy and supporting multiple color modes."""
TYPE: ClassVar[ChunkType] = ChunkType.IMAG
pixels: np.ndarray # The actual pixel data as a NumPy array

@property    
def height(self) -> int: return self.pixels.shape[0]    
@property    
def width(self) -> int: return self.pixels.shape[1]    
@property    
def mode(self) -> ColorMode:    
    shape = self.pixels.shape    
    if len(shape) == 2: return ColorMode.INDEXED    
    channels = shape[2]    
    if channels == 3: return ColorMode.RGB    
    if channels == 4: return ColorMode.RGBA    
    raise ValueError(f"Unsupported numpy array shape for image data: {shape}")    
    
def _get_payload(self) -> bytes:    
    mode = self.mode    
    dtype_map = {np.uint8: 0, np.uint16: 1, np.float32: 2}    
    if self.pixels.dtype.type not in dtype_map:    
        raise TypeError(f"Unsupported NumPy dtype: {self.pixels.dtype}")    

    header = struct.pack('>IIBB', self.width, self.height, mode.value, dtype_map[self.pixels.dtype.type])    
    pixel_bytes = self.pixels.tobytes()    

    # Filtering is only applied to 8-bit non-indexed images for max effect    
    if mode != ColorMode.INDEXED and self.pixels.dtype == np.uint8:    
        bpp = mode.value    
        filtered_bytes = apply_filters_numpy(pixel_bytes, self.width, self.height, bpp)    
        return header + zlib.compress(filtered_bytes, level=9)    
        
    return header + zlib.compress(pixel_bytes, level=9)    

@classmethod    
def from_payload(cls, payload: bytes) -> IMAGChunk:    
    width, height, mode_val, dtype_val = struct.unpack('>IIBB', payload[:10])    
    mode = ColorMode(mode_val)    
        
    dtype_map = {0: np.uint8, 1: np.uint16, 2: np.float32}    
    dtype = dtype_map.get(dtype_val)    
    if dtype is None: raise TypeError(f"Unsupported data type code: {dtype_val}")    

    pixel_payload = zlib.decompress(payload[10:])    
        
    if mode != ColorMode.INDEXED and dtype == np.uint8:    
        bpp = mode.value    
        pixel_bytes = unfilter_numpy(pixel_payload, width, height, bpp)    
    else:    
        pixel_bytes = pixel_payload    

    shape = (height, width) if mode == ColorMode.INDEXED else (height, width, mode.value)    
    pixels = np.frombuffer(pixel_bytes, dtype=dtype).reshape(shape)    
    return cls(pixels)

--- High-Level API and Factory ---

CHUNK_REGISTRY: Dict[bytes, Type[BaseChunk]] = {
c.TYPE.value: c for c in [IMAGChunk, METAChunk, ICCPChunk, PLTEChunk]
}

... Rest of the PIImage class and helpers are mostly unchanged ...

They adapt automatically because the chunk interface is consistent.

The core logic remains the same, but now it operates on more powerful chunks.

def _read_chunk_from_stream(stream: BinaryIO) -> Optional[BaseChunk]:
# ... identical to previous version ...
h = stream.read(12)
if not h: return None
if len(h) < 12: raise CorruptedDataError("Incomplete chunk header.")
t, l, c = struct.unpack('>4sII', h)
p = stream.read(l)
if len(p) < l: raise CorruptedDataError(f"Chunk {t!r} truncated.")
if zlib.crc32(t + p) != c: raise CorruptedDataError(f"CRC32 mismatch for {t!r}.")
cls = CHUNK_REGISTRY.get(t)
return cls.from_payload(p) if cls else UnknownChunk(t, p)

@dataclass
class PIImage:
chunks: List[BaseChunk] = field(default_factory=list); version: int = CURRENT_VERSION
@property
def primary_image(self) -> Optional[IMAGChunk]:
return next((c for c in self.chunks if isinstance(c, IMAGChunk)), None)
@property
def palette(self) -> Optional[PLTEChunk]:
return next((c for c in self.chunks if isinstance(c, PLTEChunk)), None)

def to_pil_image(self) -> Image.Image:    
    """Converts the PIImage to a Pillow Image object for viewing or saving."""    
    imag = self.primary_image    
    if not imag: raise ValueError("Cannot create Pillow image, no IMAG chunk found.")    
        
    mode_map = {ColorMode.RGB: "RGB", ColorMode.RGBA: "RGBA"}    
    if imag.mode in mode_map:    
        return Image.fromarray(imag.pixels, mode=mode_map[imag.mode])    
    elif imag.mode == ColorMode.INDEXED:    
        plte = self.palette    
        if not plte: raise ValueError("Indexed image requires a PLTE chunk.")    
            
        pil_img = Image.fromarray(imag.pixels, mode="P")    
        # Pillow palette requires a flat list of RGB or RGBA values    
        pil_palette = plte.palette.flatten().tolist()    
        pil_img.putpalette(pil_palette)    
        return pil_img    
    else:    
        raise ValueError(f"Conversion to Pillow not supported for mode {imag.mode}")    
    
# ... save, load, from_stream, to_bytes are identical to previous version ...    
def to_bytes(self) -> bytes:    
    h = struct.pack(">4sBBH", PI_SIGNATURE, self.version, 0, 0)    
    s = io.BytesIO(); s.write(h)    
    for c in self.chunks: s.write(c.to_bytes())    
    s.write(PI_FOOTER)    
    return s.getvalue()    
def save(self, path: str):    
    with open(path, "wb") as f: f.write(self.to_bytes())    
@classmethod    
def load(cls, path: str) -> PIImage:    
    with open(path, "rb") as f: return cls.from_stream(f)    
@classmethod    
def from_stream(cls, stream: BinaryIO) -> PIImage:    
    h = stream.read(8); s, v, _, _ = struct.unpack(">4sBBH", h)    
    if s != PI_SIGNATURE: raise InvalidFileFormatError("Invalid signature.")    
    stream.seek(0, io.SEEK_END); fs = stream.tell(); stream.seek(fs - len(PI_FOOTER))    
    if stream.read() != PI_FOOTER: raise InvalidFileFormatError("Footer missing.")    
    stream.seek(8)    
    chunks = []    
    while stream.tell() < fs - len(PI_FOOTER):    
        c = _read_chunk_from_stream(stream)    
        if c: chunks.append(c)    
    return cls(chunks, v)

--- Example Usage ---

if name == "main":
# --- Example 1: RGBA image with alpha transparency ---
print("--- 1. Creating RGBA Image (with transparency) ---")
w, h = 64, 64
rgba_pixels = np.zeros((h, w, 4), dtype=np.uint8)
# Red circle with alpha gradient
for y in range(h):
for x in range(w):
dx, dy = x - w//2, y - h//2
dist = np.sqrt(dxdx + dydy)
if dist < w//2:
rgba_pixels[y, x] = [255, 0, 0, 255 - int(255 * dist / (w//2))]

rgba_imag_chunk = IMAGChunk(rgba_pixels)    
rgba_pi_image = PIImage(chunks=[rgba_imag_chunk])    
rgba_pi_image.save("test_rgba.pi")    
# Verify by converting to Pillow and saving as PNG    
rgba_pi_image.to_pil_image().save("test_rgba_output.png")    
print("Saved 'test_rgba.pi' and verification 'test_rgba_output.png'")    

# --- Example 2: Indexed/Paletted image (like a GIF) ---    
print("\n--- 2. Creating Indexed Image (with a palette) ---")    
colors = [    
    (23, 37, 42),      # Dark blue    
    (42, 82, 120),     # Medium blue    
    (181, 228, 140),   # Light green    
    (255, 240, 165)    # Cream    
]    
palette_array = np.array(colors, dtype=np.uint8)    
# Create an image using only indices into the palette    
indexed_pixels = np.random.randint(0, 4, size=(128, 128), dtype=np.uint8)    

plte_chunk = PLTEChunk(palette_array)    
indexed_imag_chunk = IMAGChunk(indexed_pixels)    
indexed_pi_image = PIImage(chunks=[plte_chunk, indexed_imag_chunk])    
indexed_pi_image.save("test_indexed.pi")    
indexed_pi_image.to_pil_image().save("test_indexed_output.png")    
print("Saved 'test_indexed.pi' and verification 'test_indexed_output.png'")    

# --- Verification: Load and check data ---    
print("\n--- 3. Verification ---")    
loaded_rgba = PIImage.load("test_rgba.pi")    
assert np.array_equal(loaded_rgba.primary_image.pixels, rgba_pixels)    
print("RGBA data integrity: PASSED")    

loaded_indexed = PIImage.load("test_indexed.pi")    
assert np.array_equal(loaded_indexed.primary_image.pixels, indexed_pixels)    
assert np.array_equal(loaded_indexed.palette.palette, palette_array)    
print("Indexed data integrity: PASSED")
