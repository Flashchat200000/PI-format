# -*- coding: utf-8 -*-
"""
PI (Power Image) Format - An extensible, chunk-based image format implementation.

This module provides a framework for creating, manipulating, and parsing
files of the PI image format. It is designed with extensibility and robustness
in mind, featuring CRC32 checksums for data integrity and a class-based
chunk system that is easy to expand.
"""
from __future__ import annotations

import struct
import zlib
import io
from dataclasses import dataclass, field
from enum import Enum
from typing import BinaryIO, Dict, List, Optional, Type, Union

# --- Core Format Constants ---

PI_SIGNATURE = b'PI2F'
CURRENT_VERSION = 0x02
PI_FOOTER = b'PI_E'


# --- Enumerations for Type Safety ---

class ChunkType(Enum):
    """Enumeration of all official chunk types."""
    IMAG = b'IMAG'  # Image Data Chunk
    META = b'META'  # Metadata Chunk
    ICCP = b'ICCP'  # ICC Color Profile Chunk
    # Future chunk types can be added here, e.g., ANIM = b'ANIM'

class Compression(Enum):
    """Enumeration of supported compression types."""
    NONE = 0
    ZLIB = 1


# --- Custom Exception Hierarchy ---

class PIException(Exception):
    """Base class for all exceptions related to the PI format."""
    pass

class InvalidFileFormatError(PIException):
    """Raised when a file does not conform to the PI format specification."""
    pass

class CorruptedDataError(PIException):
    """Raised when data integrity checks (like CRC32) fail."""
    pass


# --- Chunk Definitions using Dataclasses ---

@dataclass(frozen=True)
class BaseChunk:
    """Abstract base class for all chunk types. frozen=True makes instances immutable."""
    TYPE: ClassVar[ChunkType]

    def to_bytes(self) -> bytes:
        """Serializes the entire chunk (header + payload) into bytes."""
        payload = self._get_payload()
        # CRC32 is calculated over the chunk type and its payload.
        crc = zlib.crc32(self.TYPE.value + payload)
        header = struct.pack('>4sII', self.TYPE.value, len(payload), crc)
        return header + payload

    def _get_payload(self) -> bytes:
        """
        Internal method to get the chunk's payload.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError

    @classmethod
    def from_payload(cls, payload: bytes) -> BaseChunk:
        """
        Class method to create a chunk instance from its payload.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError

@dataclass(frozen=True)
class IMAGChunk(BaseChunk):
    """Chunk containing the primary pixel data of an image."""
    TYPE: ClassVar[ChunkType] = ChunkType.IMAG

    width: int
    height: int
    pixel_data: bytes
    depth: int = 8
    channels: int = 3
    compression: Compression = Compression.ZLIB

    def _get_payload(self) -> bytes:
        data_header = struct.pack('>IIBBB', self.width, self.height, self.depth, self.channels, self.compression.value)
        
        if self.compression == Compression.ZLIB:
            compressed_pixels = zlib.compress(self.pixel_data, level=9)
            return data_header + compressed_pixels
        return data_header + self.pixel_data

    @classmethod
    def from_payload(cls, payload: bytes) -> IMAGChunk:
        width, height, depth, channels, compression_val = struct.unpack('>IIBBB', payload[:11])
        pixel_payload = payload[11:]

        compression = Compression(compression_val)
        if compression == Compression.ZLIB:
            pixel_data = zlib.decompress(pixel_payload)
        else:
            pixel_data = pixel_payload

        return cls(width, height, pixel_data, depth, channels, compression)

@dataclass(frozen=True)
class METAChunk(BaseChunk):
    """Chunk for storing key-value string metadata."""
    TYPE: ClassVar[ChunkType] = ChunkType.META
    metadata: Dict[str, str] = field(default_factory=dict)

    def _get_payload(self) -> bytes:
        # Simple UTF-8 encoded key-value pairs with length prefixes.
        payload = io.BytesIO()
        for key, value in self.metadata.items():
            key_bytes = key.encode('utf-8')
            value_bytes = value.encode('utf-8')
            payload.write(struct.pack('>I', len(key_bytes)))
            payload.write(key_bytes)
            payload.write(struct.pack('>I', len(value_bytes)))
            payload.write(value_bytes)
        return payload.getvalue()

    @classmethod
    def from_payload(cls, payload: bytes) -> METAChunk:
        metadata = {}
        stream = io.BytesIO(payload)
        while stream.tell() < len(payload):
            key_len = struct.unpack('>I', stream.read(4))[0]
            key = stream.read(key_len).decode('utf-8')
            value_len = struct.unpack('>I', stream.read(4))[0]
            value = stream.read(value_len).decode('utf-8')
            metadata[key] = value
        return cls(metadata)

@dataclass(frozen=True)
class ICCPChunk(BaseChunk):
    """Chunk for embedding an ICC color profile."""
    TYPE: ClassVar[ChunkType] = ChunkType.ICCP
    profile_data: bytes
    
    def _get_payload(self) -> bytes:
        # The payload is simply the raw ICC profile data.
        return self.profile_data

    @classmethod
    def from_payload(cls, payload: bytes) -> ICCPChunk:
        return cls(profile_data=payload)

@dataclass(frozen=True)
class UnknownChunk(BaseChunk):
    """A placeholder for chunks of an unknown type, allowing forward compatibility."""
    chunk_type_bytes: bytes
    payload: bytes

    @property
    def TYPE(self) -> None:
        raise NotImplementedError("UnknownChunk does not have a static TYPE.")

    def to_bytes(self) -> bytes:
        crc = zlib.crc32(self.chunk_type_bytes + self.payload)
        header = struct.pack('>4sII', self.chunk_type_bytes, len(self.payload), crc)
        return header + self.payload


# --- Chunk Factory and Registry ---
CHUNK_REGISTRY: Dict[bytes, Type[BaseChunk]] = {
    ChunkType.IMAG.value: IMAGChunk,
    ChunkType.META.value: METAChunk,
    ChunkType.ICCP.value: ICCPChunk,
}

def _read_chunk_from_stream(stream: BinaryIO) -> Optional[BaseChunk]:
    """Reads a single chunk from a stream and returns a parsed chunk object."""
    chunk_header_bytes = stream.read(12) # 4sII = 12 bytes
    if not chunk_header_bytes:
        return None
    if len(chunk_header_bytes) < 12:
        raise CorruptedDataError("Incomplete chunk header found.")

    chunk_type_bytes, length, expected_crc = struct.unpack('>4sII', chunk_header_bytes)
    payload = stream.read(length)
    if len(payload) < length:
        raise CorruptedDataError(f"Chunk {chunk_type_bytes!r} is truncated.")

    actual_crc = zlib.crc32(chunk_type_bytes + payload)
    if actual_crc != expected_crc:
        raise CorruptedDataError(f"CRC32 mismatch for chunk {chunk_type_bytes!r}. Data is corrupt.")

    chunk_class = CHUNK_REGISTRY.get(chunk_type_bytes)
    if chunk_class:
        return chunk_class.from_payload(payload)
    
    # If the chunk type is not in the registry, treat it as an UnknownChunk.
    return UnknownChunk(chunk_type_bytes, payload)


# --- High-Level API: The PIImage Class ---

@dataclass
class PIImage:
    """
    A high-level container for a PI image, composed of a list of chunks.
    Provides methods for loading, saving, and accessing image data.
    """
    chunks: List[BaseChunk] = field(default_factory=list)
    version: int = CURRENT_VERSION
    flags: int = 0

    @property
    def primary_image(self) -> Optional[IMAGChunk]:
        """Convenience property to get the first IMAG chunk."""
        for chunk in self.chunks:
            if isinstance(chunk, IMAGChunk):
                return chunk
        return None

    @property
    def metadata(self) -> Dict[str, str]:
        """Convenience property to get metadata from the first META chunk."""
        for chunk in self.chunks:
            if isinstance(chunk, METAChunk):
                return chunk.metadata
        return {}
    
    def get_chunks_by_type(self, chunk_type: Union[ChunkType, Type[BaseChunk]]) -> List[BaseChunk]:
        """Returns all chunks of a specific type."""
        if isinstance(chunk_type, ChunkType):
            return [c for c in self.chunks if hasattr(c, 'TYPE') and c.TYPE == chunk_type]
        else: # Assumes a class type like IMAGChunk
             return [c for c in self.chunks if isinstance(c, chunk_type)]

    def to_bytes(self) -> bytes:
        """Serializes the entire PIImage object into a byte string."""
        header = struct.pack(">4sBBH", PI_SIGNATURE, self.version, self.flags, 0) # 8 bytes total
        
        stream = io.BytesIO()
        stream.write(header)
        for chunk in self.chunks:
            stream.write(chunk.to_bytes())
        stream.write(PI_FOOTER)
        
        return stream.getvalue()

    def save(self, path: str):
        """Saves the PIImage to a file."""
        with open(path, "wb") as f:
            f.write(self.to_bytes())

    @classmethod
    def from_stream(cls, stream: BinaryIO) -> PIImage:
        """Loads a PIImage from a binary stream."""
        header_bytes = stream.read(8)
        if len(header_bytes) < 8:
            raise InvalidFileFormatError("File is too small to be a valid PI image.")
        
        signature, version, flags, _ = struct.unpack(">4sBBH", header_bytes)
        if signature != PI_SIGNATURE:
            raise InvalidFileFormatError(f"Invalid signature. Expected {PI_SIGNATURE!r}, got {signature!r}.")

        # Read chunks until the footer
        # We must read the file to find the footer position
        stream.seek(0, io.SEEK_END)
        file_size = stream.tell()
        stream.seek(file_size - len(PI_FOOTER))
        if stream.read() != PI_FOOTER:
             raise InvalidFileFormatError("File footer is missing or corrupt.")
        
        stream.seek(8) # Seek back to after the header
        chunks = []
        while stream.tell() < file_size - len(PI_FOOTER):
            chunk = _read_chunk_from_stream(stream)
            if chunk:
                chunks.append(chunk)
            else:
                break
        
        return cls(chunks, version, flags)

    @classmethod
    def load(cls, path: str) -> PIImage:
        """Loads a PIImage from a file path."""
        with open(path, "rb") as f:
            return cls.from_stream(f)


# --- Example Usage ---
if __name__ == "__main__":
    print("--- 1. Creating a new PI Image object ---")
    width, height = 32, 32
    # Create a simple green pixel buffer
    pixel_buffer = b'\x00\xFF\x00' * (width * height)

    # Create the core image data chunk
    imag_chunk = IMAGChunk(width, height, pixel_buffer)
    
    # Create a metadata chunk
    meta_chunk = METAChunk({
        "Author": "Professional AI",
        "Software": "PI Format Library v2.0",
        "Description": "A test image demonstrating format capabilities."
    })
    
    # Create an ICC profile chunk (using placeholder data)
    # In a real scenario, you would load an actual .icc file's content here.
    icc_placeholder_data = b"This is where the binary ICC profile would go."
    iccp_chunk = ICCPChunk(icc_placeholder_data)
    
    # Assemble the PIImage
    pi_image = PIImage(chunks=[imag_chunk, meta_chunk, iccp_chunk])
    print(f"PIImage created with {len(pi_image.chunks)} chunks.")

    # --- 2. Saving the image to a file ---
    FILE_PATH = "test_image.pi"
    try:
        pi_image.save(FILE_PATH)
        print(f"Image successfully saved to '{FILE_PATH}'")
    except Exception as e:
        print(f"Error saving file: {e}")

    # --- 3. Loading the image from the file ---
    print("\n--- 3. Loading image from file ---")
    try:
        loaded_image = PIImage.load(FILE_PATH)
        print("Image successfully loaded.")
        
        # --- 4. Inspecting the loaded data ---
        print("\n--- 4. Inspecting loaded image data ---")
        
        # Accessing the primary image dimensions
        img_info = loaded_image.primary_image
        if img_info:
            print(f"Image dimensions: {img_info.width}x{img_info.height}")
            print(f"Pixel data size: {len(img_info.pixel_data)} bytes")
            assert img_info.pixel_data == pixel_buffer, "Pixel data integrity check PASSED."
        
        # Accessing metadata
        metadata = loaded_image.metadata
        print("\nMetadata found:")
        for key, val in metadata.items():
            print(f"  - {key}: {val}")
        
        # Accessing other chunks by type
        iccp_chunks_found = loaded_image.get_chunks_by_type(ICCPChunk)
        if iccp_chunks_found:
            print(f"\nFound {len(iccp_chunks_found)} ICCP chunk(s).")
            # print(f"  Profile data (first 16 bytes): {iccp_chunks_found[0].profile_data[:16]}...")

    except PIException as e:
        print(f"A PI Format error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
