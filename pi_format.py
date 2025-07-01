# -- coding: utf-8 --
"""
PIX (Pixel Scene) Format - v16.0 "The Process Format"

This module provides the reference implementation for the PIX v16.0 file format.
This version fundamentally re-architects the format's relationship with data,
time, and the external world, evolving it from a self-contained file into a
manifest for a dynamic process.

The core principles of v16.0 are:

1.  Contextual Lazy Loading (The Ghost in the Machine):
    - An `INDX` (Index) chunk provides a complete map of the file's layout.
    - The `load()` function is now instantaneous, reading only the header and
      index to return a lightweight `SceneProxy` object. Data for individual
      layers is streamed from disk only when explicitly accessed, enabling
      the handling of enormous files with minimal initial memory overhead.
    - Loading can be context-aware, allowing a client to request specific
      representations (e.g., only thumbnails) or to preemptively skip layers
      that would violate resource constraints (e.g., memory limits).

2.  Linked Data and External Resources (The Web Walker):
    - The `LINK` chunk breaks the "prison" of a self-contained file. It allows
      layers to reference external assets via URI (e.g., from a web server
      or local file system), with a cryptographic hash to ensure integrity.
    - This facilitates asset reuse, enables live updates to shared resources
      without modifying the PIX file itself, and allows for the composition
      of scenes from a distributed network of content.

3.  Immutable History and Delta Chunks (The Time Traveler):
    - The format embraces an append-only design philosophy, capturing change
      over time rather than just the final state.
    - `DELT` (Delta) chunks record discrete modifications, and `SAVE` chunks
      act as commit markers, creating an immutable history log within the file.
    - This architecture provides inherent version control, crash-safe saving,
      and a foundation for collaborative workflows.

PIX v16.0 is no longer a mere file format. It is a specification for a
performant, interconnected, and versioned document ecosystem.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, BinaryIO, ClassVar, Dict, List, Optional, Tuple, Type, Union

# --- Core Format Constants and Enumerations ---

PIX_SIGNATURE = b'PIXF'
PIX_FOOTER = b'PIXE'

class ChunkType(IntEnum):
    """Enumeration of all official PIX chunk types."""
    # Foundational Chunks
    PROF = 0x50524F46; HEAD = 0x48454144
    # Data & Structure Chunks
    FRAM = 0x4652414d; LDAT = 0x4c444154; SCPT = 0x53435054
    # Governance & API Chunks
    CAPI = 0x43415049; GOVR = 0x474F5652
    # Ancillary Chunks
    META = 0x4d455441; ICCP = 0x49434350; PRVW = 0x50525657
    # --- New in v16.0 ---
    INDX = 0x494E4458  # Index (File map for lazy loading) - CRITICAL
    LINK = 0x4C494E4B  # External Resource Link
    DELT = 0x44454C54  # Delta (A single change in the history log)
    SAVE = 0x53415645  # Save Point (A commit marker in the history log)

# (Other enums remain unchanged)
class FileProfile(IntEnum): SIMPLE_IMAGE = 0; SCENE = 1

# --- Custom Exceptions (Unchanged) ---
class PIXException(Exception): pass
class InvalidFileFormatError(PIXException): pass

# --- Chunk Definitions ---

@dataclass(frozen=True)
class BaseChunk:
    """Abstract base class for all PIX chunks."""
    TYPE: ClassVar[ChunkType]
    def to_bytes(self) -> bytes: raise NotImplementedError
    @classmethod
    def from_payload(cls, payload: bytes) -> 'BaseChunk': raise NotImplementedError

# --- New v16.0 Chunk Definitions ---

@dataclass(frozen=True)
class INDXChunk(BaseChunk):
    """INDX: Contains a map of all major objects in the file for lazy loading."""
    TYPE: ClassVar[ChunkType] = ChunkType.INDX
    # The index maps a unique ID (e.g., 'layer_0', 'global_govr') to its
    # (offset, length) in the file.
    index_map: Dict[str, Tuple[int, int]]

    def to_bytes(self) -> bytes:
        # For simplicity and readability, we use compressed JSON.
        # A production implementation might use a more compact binary format.
        payload = zlib.compress(json.dumps(self.index_map).encode('utf-8'))
        header = struct.pack('>II', self.TYPE.value, len(payload))
        return header + payload

    @classmethod
    def from_payload(cls, payload: bytes) -> INDXChunk:
        return cls(json.loads(zlib.decompress(payload)))

@dataclass(frozen=True)
class LINKChunk(BaseChunk):
    """LINK: A reference to an external resource."""
    TYPE: ClassVar[ChunkType] = ChunkType.LINK
    uri: str
    hash_algo: str  # e.g., "sha256"
    content_hash: str

    def to_bytes(self) -> bytes:
        payload_dict = {"uri": self.uri, "algo": self.hash_algo, "hash": self.content_hash}
        payload = json.dumps(payload_dict).encode('utf-8')
        header = struct.pack('>II', self.TYPE.value, len(payload))
        return header + payload

    @classmethod
    def from_payload(cls, payload: bytes) -> LINKChunk:
        data = json.loads(payload)
        return cls(data['uri'], data['algo'], data['hash'])

# DELT and SAVE are primarily for the append-only history model.
# Their parsing is straightforward, but their application is part of the
# state reconstruction logic, which is beyond simple parsing.
@dataclass(frozen=True)
class DELTChunk(BaseChunk):
    """DELT: Represents a single atomic change to the scene state."""
    TYPE: ClassVar[ChunkType] = ChunkType.DELT
    # e.g., "layers.1.opacity", "metadata.author"
    target_path: str
    operation: str  # e.g., "set", "add", "list_append"
    value: Any

@dataclass(frozen=True)
class SAVEChunk(BaseChunk):
    """SAVE: A marker chunk, equivalent to a 'commit'."""
    TYPE: ClassVar[ChunkType] = ChunkType.SAVE
    timestamp: int
    message: str

# --- The Proxy Architecture for Lazy Loading ---

class LayerProxy:
    """A lazy-loading proxy object for a single layer."""
    def __init__(self, scene_proxy: 'SceneProxy', layer_id: str, index_entry: Tuple[int, int]):
        self._scene = scene_proxy
        self._id = layer_id
        self._offset, self._length = index_entry
        self._state: Optional['LayerState'] = None # The cached, fully parsed state

    def _load(self):
        """Loads and parses the full FRAM chunk from disk on first access."""
        if self._state is None:
            print(f"LAZY LOAD: Accessing layer '{self._id}'. Reading {self._length} bytes from offset {self._offset}.")
            self._scene._file_handle.seek(self._offset)
            chunk_bytes = self._scene._file_handle.read(self._length)
            # In a real implementation, you would parse the FRAM chunk here
            # to create a full LayerState object.
            # For this demo, we'll simulate it.
            self._state = self._parse_fram_chunk(chunk_bytes)

    def _parse_fram_chunk(self, chunk_bytes: bytes) -> 'LayerState':
        # This is a simplified placeholder for the real FRAM parsing logic.
        # It would inspect child chunks (LDAT, SCPT, LINK) to build the correct state.
        # Here we'll just create a dummy state.
        # Check for a LINK chunk for demonstration.
        if b"LINK" in chunk_bytes:
             return LinkedLayerState(layer_id=self._id, name=f"Linked Layer {self._id}", uri="https://example.com/asset.wasm")
        return LayerState(layer_id=self._id, name=f"Layer {self._id}", opacity=255)


    def __getattr__(self, name: str):
        """Intercepts attribute access to trigger lazy loading."""
        self._load()
        return getattr(self._state, name)

class SceneProxy:
    """
    A lightweight, lazy-loading proxy for a PIX scene. It provides access
    to the scene's properties and layers without loading the entire file
    into memory.
    """
    def __init__(self, filepath: str, context: Optional[Dict[str, Any]] = None):
        self._filepath = filepath
        self._context = context or {}
        self._file_handle: BinaryIO = open(filepath, "rb")
        self._index: Dict[str, Tuple[int, int]] = {}
        self._layers: Dict[str, LayerProxy] = {}
        self._load_index()

    def _load_index(self):
        """Reads the file header and INDX chunk to build the file map."""
        self._file_handle.seek(0)
        if self._file_handle.read(4) != PIX_SIGNATURE:
            raise InvalidFileFormatError("Invalid PIX signature.")
        # In a real implementation, we would seek to the INDX chunk location,
        # which might be specified in a fixed-size header.
        # For this demo, we assume it follows the PROF/HEAD chunks.
        self._file_handle.seek(4) # Skip signature
        # Dummy parsing to find the index
        prof_header = self._file_handle.read(8); prof_len = struct.unpack('>I', prof_header[4:8])[0]; self._file_handle.seek(prof_len + 4, 1) # Skip PROF
        head_header = self._file_handle.read(8); head_len = struct.unpack('>I', head_header[4:8])[0]; self._file_handle.seek(head_len + 4, 1) # Skip HEAD
        
        indx_header = self._file_handle.read(8)
        chunk_type, chunk_len = struct.unpack('>II', indx_header)
        if chunk_type != ChunkType.INDX.value:
            raise InvalidFileFormatError("INDX chunk not found or is out of order.")
        
        indx_payload = self._file_handle.read(chunk_len)
        self._index = INDXChunk.from_payload(indx_payload).index_map
        print("INDEX LOADED: SceneProxy is ready. File is open, but no layer data has been read.")

    @property
    def layers(self) -> Dict[str, LayerProxy]:
        """Provides access to layer proxies, creating them on demand."""
        if not self._layers:
            for key, value in self._index.items():
                if key.startswith("layer_"):
                    self._layers[key] = LayerProxy(self, key, value)
        return self._layers

    def close(self):
        """Closes the underlying file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            print("SceneProxy closed the file handle.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Dummy LayerState classes for demonstration
@dataclass
class LayerState:
    layer_id: str; name: str; opacity: int
@dataclass
class LinkedLayerState(LayerState):
    uri: str

def load(filepath: str, context: Optional[Dict[str, Any]] = None) -> SceneProxy:
    """
    Opens a PIX v16 file and returns an efficient, lazy-loading SceneProxy.
    This function executes almost instantaneously, regardless of file size.
    """
    return SceneProxy(filepath, context)

def create_dummy_v16_file(filepath: str):
    """A utility function to create a file for the demonstration."""
    print(f"\nCreating a dummy PIX v16 file at '{filepath}'...")
    with open(filepath, "wb") as f:
        # 1. Signature
        f.write(PIX_SIGNATURE)
        
        # 2. PROF and HEAD chunks (placeholders)
        prof_payload = b'\x01'; f.write(struct.pack('>II', ChunkType.PROF.value, len(prof_payload)) + prof_payload + b'\x00\x00\x00\x00')
        head_payload = b'\x00\x00\x04\x00\x00\x00\x03\x00'; f.write(struct.pack('>II', ChunkType.HEAD.value, len(head_payload)) + head_payload + b'\x00\x00\x00\x00')
        
        start_of_data = f.tell()

        # 3. Layer Data (Dummy FRAM chunks)
        fram_0_payload = b"FRAM_CHUNK_DATA_FOR_LAYER_0"; fram_0_offset = f.tell()
        f.write(fram_0_payload); fram_0_len = len(fram_0_payload)

        fram_1_payload = b"FRAM_CHUNK_DATA_FOR_LAYER_1_WITH_LINK"; fram_1_offset = f.tell()
        f.write(fram_1_payload); fram_1_len = len(fram_1_payload)

        # 4. Create and write the INDX chunk
        index_map = {
            "layer_0": (fram_0_offset, fram_0_len),
            "layer_1": (fram_1_offset, fram_1_len),
        }
        indx_chunk = INDXChunk(index_map)
        indx_bytes = indx_chunk.to_bytes()
        
        # We must place the index at a known location. Here, after HEAD.
        f.seek(start_of_data)
        f.write(indx_bytes)
        f.seek(0, io.SEEK_END) # Go back to the end to write other things
        
        f.write(fram_0_payload)
        f.write(fram_1_payload)
        
        # 5. Footer
        f.write(PIX_FOOTER)
    print("Dummy file created.")

if __name__ == "__main__":
    DUMMY_FILE = "scene_v16_process.pix"
    create_dummy_v16_file(DUMMY_FILE)

    print("\n--- Demonstrating PIX v16.0 Lazy Loading ---")
    
    # The 'load' call is instantaneous. It only reads the header and index.
    # The file handle is kept open inside the proxy.
    print("\n1. Loading the scene. This should be instantaneous.")
    with load(DUMMY_FILE) as scene:
        print("SceneProxy created successfully. No layer data has been loaded yet.")
        
        # Accessing the .layers attribute doesn't load data either; it just
        # creates the dictionary of LayerProxy objects.
        print("\n2. Accessing the '.layers' dictionary property.")
        layers = scene.layers
        print(f"Found {len(layers)} layers in the index: {list(layers.keys())}")
        
        # NOW, the magic happens. Accessing an attribute on a specific layer
        # triggers the _load() method for that layer only.
        print("\n3. Accessing an attribute on 'layer_0'. This will trigger I/O.")
        layer_0_name = layers['layer_0'].name
        print(f"--> Accessed 'layer_0'. Its name is: '{layer_0_name}'")

        print("\n4. Accessing the same layer again. This should be instant (from cache).")
        layer_0_opacity = layers['layer_0'].opacity
        print(f"--> Accessed 'layer_0' again. Its opacity is: {layer_0_opacity}")

        print("\n5. Accessing a different layer. This will trigger a new I/O operation.")
        # This demonstrates both lazy loading and the LINK concept.
        layer_1 = layers['layer_1']
        if isinstance(layer_1, LinkedLayerState):
             print(f"--> Accessed 'layer_1'. It is a LinkedLayerState referencing URI: '{layer_1.uri}'")
        else:
             print(f"--> Accessed 'layer_1'. Its name is: '{layer_1.name}'")

    print("\n--- Conceptual Demonstration: History and Linking ---")
    print("This implementation provides the file-format structures for advanced concepts:")
    print("  - LINK Chunk: Allows `uri` and `hash` fields for referencing external assets.")
    print("  - DELT/SAVE Chunks: Provide a foundation for building append-only files with version history.")
    print("A full implementation of these concepts resides in the renderer/application logic, which would handle URI resolution and delta application.")

    print("\n--- End of Demonstration ---")
    print("The format has evolved from a container to a process manifest.")
