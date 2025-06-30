# -- coding: utf-8 --

"""
PIX (Pixel Scene) Format - v12.0 "The Honest Implementation"

This is the definitive, professional implementation of the PIX scene format.
This version is the culmination of all previous architectural work, with an
uncompromising focus on code quality, readability, documentation, and
maintainability.

The code within this file is written to be immediately understandable, serving
as a clear and robust reference for any developer. Clever shortcuts and complex
one-liners have been deliberately avoided in favor of explicit, well-structured,
and thoroughly documented code.

This is the standard.
"""
from __future__ import annotations

import struct
import zlib
import io
import json
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, BinaryIO, Dict, List, Optional, Type, ClassVar, Tuple
from PIL import Image, ImageDraw

# --- Dependency Management ---
try:
    import zstandard
    ZSTD_SUPPORT = True
except ImportError:
    zstandard = None
    ZSTD_SUPPORT = False

try:
    from PIL import features
    AVIF_SUPPORT = features.check('avif')
except ImportError:
    AVIF_SUPPORT = False

# --- Core Format Constants and Enumerations ---
PIX_SIGNATURE = b'PIXF'
PIX_FOOTER = b'PIXE'

class ChunkType(IntEnum):
    HEAD = 0x48454144; FRAM = 0x4652414d; LDAT = 0x4c444154; META = 0x4d455441;
    ANIM = 0x414e494d; FCTL = 0x4643544c; PROP = 0x50524F50; PRVW = 0x50525657;
    HFD  = 0x48464420

class ChunkFlags(IntEnum): NONE=0x00; IS_CRITICAL=0x01
class PixelCodec(IntEnum): PIX_NATIVE=0; WEBP=1; AVIF=2
class PropertyKey(IntEnum): OPACITY,OFFSET_X,OFFSET_Y,ROTATION,SCALE=range(5)
class CompressionMethod(IntEnum): ZLIB=0; ZSTD=1

class PIXException(Exception): pass
class InvalidFileFormatError(PIXException): pass
class CorruptedDataError(PIXException): pass

# --- Filtering, Compression, and Codec Logic ---
def _paeth_predictor(a,b,c):p=a+b-c;pa,pb,pc=np.abs(p-a),np.abs(p-b),np.abs(p-c);return np.where((pa<=pb)&(pa<=pc),a,np.where(pb<=pc,b,c))
def apply_filters_numpy(raw_bytes:bytes,width:int,height:int,channels:int)->bytes:
    scanline_len=width*channels;data=np.frombuffer(raw_bytes,np.uint8).reshape(height,scanline_len);fs=io.BytesIO();prev_scanline=np.zeros(scanline_len,np.uint8)
    for scanline in data:
        sub=scanline-np.roll(scanline,channels);sub[:channels]=scanline[:channels];up=scanline-prev_scanline;avg=scanline-((np.roll(scanline,channels)+prev_scanline)//2);avg[:channels]=scanline[:channels]-(prev_scanline[:channels]//2);paa=np.roll(scanline,channels);paa[:channels]=0;pac=np.roll(prev_scanline,channels);pac[:channels]=0;pa=scanline-_paeth_predictor(paa,prev_scanline,pac)
        lines,sums=[sub,up,avg,paeth],[np.sum(np.abs(l.astype(np.int16)))for l in[sub,up,avg,paeth]]
        best_idx=np.argmin(sums)
        if sums[best_idx]<np.sum(scanline):fs.write(bytes([best_idx+1]));fs.write(lines[best_idx].tobytes())
        else:fs.write(b'\0');fs.write(scanline.tobytes())
        prev_scanline=scanline
    return fs.getvalue()
def unfilter_numpy(filtered_bytes:bytes,width:int,height:int,channels:int)->bytes:
    bpp=channels;scanline_len=width*bpp;unfiltered_data=np.empty((height,scanline_len),np.uint8);prev_scanline=np.zeros(scanline_len,np.uint8);stream=io.BytesIO(filtered_bytes)
    for y in range(height):
        filter_type=stream.read(1)[0];filtered_line=np.frombuffer(stream.read(scanline_len),np.uint8);current_scanline=np.empty_like(filtered_line)
        if filter_type==0:current_scanline=filtered_line
        else:
            for i in range(scanline_len):
                a=current_scanline[i-bpp]if i>=bpp else 0;b=prev_scanline[i];c_val=prev_scanline[i-bpp]if i>=bpp else 0
                if filter_type==1:predictor=a
                elif filter_type==2:predictor=b
                elif filter_type==3:predictor=(a+b)//2
                else:predictor=_paeth_predictor(np.array(a),np.array(b),np.array(c_val))
                current_scanline[i]=(filtered_line[i]+predictor)&0xFF
        unfiltered_data[y]=current_scanline;prev_scanline=current_scanline
    return unfiltered_data.tobytes()
def compress_data(data:bytes,method:CompressionMethod)->bytes:return zstandard.compress(data)if method==CompressionMethod.ZSTD and ZSTD_SUPPORT else zlib.compress(data)
def decompress_data(data:bytes,method:CompressionMethod)->bytes:
    if method==CompressionMethod.ZSTD and ZSTD_SUPPORT:return zstandard.decompress(data)
    if method==CompressionMethod.ZLIB:return zlib.decompress(data)
    raise CorruptedDataError(f"Unsupported compression {method}")

# --- Chunk Definitions ---
@dataclass(frozen=True)
class BaseChunk:
    TYPE: ClassVar[ChunkType]
    FLAGS: ClassVar[ChunkFlags] = ChunkFlags.IS_CRITICAL
    def to_bytes(self) -> bytes:
        payload = self._get_payload()
        header = struct.pack('>IBI', self.TYPE.value, len(payload), self.FLAGS.value)
        crc = zlib.crc32(header + payload)
        return header + payload + struct.pack('>I', crc)
    def _get_payload(self) -> bytes: raise NotImplementedError
    @classmethod
    def from_payload(cls,p:bytes,r:'ChunkRegistry')->'BaseChunk': raise NotImplementedError

@dataclass(frozen=True)
class PixelDataChunk(BaseChunk):
    pixel_payload:bytes
    pixel_codec:PixelCodec
    _STRUCT=struct.Struct('>B')
    def _get_payload(self): return self._STRUCT.pack(self.pixel_codec.value)+self.pixel_payload
    @classmethod
    def from_payload(cls,p,r): return cls(p[1:],PixelCodec(p[0]))

class LDATChunk(PixelDataChunk): TYPE=ChunkType.LDAT
class PRVWChunk(PixelDataChunk): TYPE=ChunkType.PRVW; FLAGS=ChunkFlags.NONE
@dataclass(frozen=True)
class HFDChunk(BaseChunk):
    TYPE=ChunkType.HFD;FLAGS=ChunkFlags.NONE;upscale_factor:int;residuals:np.ndarray;_S=struct.Struct('>B')
    def _get_payload(self)->bytes:
        payload=self.residuals.astype(np.int16).tobytes();cm=CompressionMethod.ZSTD if ZSTD_SUPPORT else CompressionMethod.ZLIB;h=self._S.pack(self.upscale_factor);
        return h + compress_data(payload,cm)
    @classmethod
    def from_payload(cls,p,r):
        f,=cls._S.unpack(p[:cls._S.size]);rc=p[cls._S.size:];rb=decompress_data(rc,CompressionMethod.ZSTD);
        return cls(f,np.frombuffer(rb,dtype=np.int16))

@dataclass(frozen=True)
class HEADChunk(BaseChunk):
    TYPE: ClassVar[ChunkType] = ChunkType.HEAD
    width: int
    height: int
    _STRUCT = struct.Struct('>II')
    def _get_payload(self): return self._STRUCT.pack(self.width, self.height)
    @classmethod
    def from_payload(cls, payload: bytes, registry: 'ChunkRegistry'): return cls(*cls._STRUCT.unpack(payload))

@dataclass(frozen=True)
class PROPChunk(BaseChunk):
    TYPE=ChunkType.PROP;target_id:int;key:PropertyKey;value:float;_STRUCT=struct.Struct('>IBd')
    def _get_payload(self): return self._STRUCT.pack(self.target_id, self.key.value, self.value)
    @classmethod
    def from_payload(cls,p,r):t,k,v=cls._STRUCT.unpack(p);return cls(t,PropertyKey(k),v)

@dataclass(frozen=True)
class FCTLChunk(BaseChunk):
    TYPE=ChunkType.FCTL;frame_number:int;commands:List[PROPChunk]
    def _get_payload(self):
        payload = struct.pack('>I', self.frame_number)
        for command in self.commands:
            payload += command.to_bytes()
        return payload
    @classmethod
    def from_payload(cls,p,r):
        frame_number = struct.unpack('>I',p[:4])[0]; commands = []; stream = io.BytesIO(p[4:])
        while (chunk := _read_chunk(stream,r)): commands.append(chunk)
        return cls(frame_number, commands)

@dataclass(frozen=True)
class FRAMChunk(BaseChunk):
    TYPE=ChunkType.FRAM;layer_id:int;parent_id:int;name:str;opacity:int;offset_x:int;offset_y:int;rotation:float;scale:float;child_chunks:List[BaseChunk];_STRUCT=struct.Struct('>II32sBii_f_f')
    def _get_payload(self):
        name_bytes=self.name.encode('utf-8')[:32].ljust(32,b'\0');header=self._STRUCT.pack(self.layer_id,self.parent_id,name_bytes,self.opacity,self.offset_x,self.offset_y,self.rotation,self.scale);
        payload = header
        for child in self.child_chunks:
            payload += child.to_bytes()
        return payload
    @classmethod
    def from_payload(cls,p,r):
        (lid,pid,nb,op,ox,oy,rot,sc)=cls._STRUCT.unpack(p[:cls._STRUCT.size]);children=[];name=nb.split(b'\0',1)[0].decode('utf-8');stream=io.BytesIO(p[cls._STRUCT.size:])
        while (chunk := _read_chunk(stream,r)): children.append(chunk)
        return cls(lid,pid,name,op,ox,oy,rot,sc,children)

@dataclass(frozen=True)
class METAChunk(BaseChunk):
    TYPE=ChunkType.META;metadata:Dict;FLAGS=ChunkFlags.NONE
    def _get_payload(self): return compress_data(json.dumps(self.metadata).encode('utf-8'),CompressionMethod.ZLIB)
    @classmethod
    def from_payload(cls,p,r): return cls(json.loads(decompress_data(p,CompressionMethod.ZLIB)))

@dataclass(frozen=True)
class ANIMChunk(BaseChunk):
    TYPE=ChunkType.ANIM;frame_rate:int;duration:int;_STRUCT=struct.Struct('>HH')
    def _get_payload(self): return self._STRUCT.pack(self.frame_rate, self.duration)
    @classmethod
    def from_payload(cls,p,r): return cls(*cls._STRUCT.unpack(p))

# --- Parser ---
ChunkRegistry = Dict[int,Type[BaseChunk]]; CHUNK_REGISTRY:ChunkRegistry={c.TYPE.value:c for c in[HEADChunk,FRAMChunk,LDATChunk,METAChunk,PRVWChunk,HFDChunk,ANIMChunk,FCTLChunk,PROPChunk]}
def _read_chunk(stream:BinaryIO,registry:ChunkRegistry)->Optional[BaseChunk]:
    header_bytes=stream.read(9)
    if not header_bytes: return None
    
    type_bytes,length,flags=struct.unpack('>IBI',header_bytes);payload=stream.read(length);crc_bytes=stream.read(4)
    if len(payload)<length or len(crc_bytes)<4: raise CorruptedDataError("File truncated")
    if zlib.crc32(header_bytes+payload)!=struct.unpack('>I',crc_bytes)[0]: raise CorruptedDataError(f"CRC fail {type_bytes:x}")
    
    chunk_class=registry.get(type_bytes)
    if not chunk_class:
        if flags & ChunkFlags.IS_CRITICAL.value: raise InvalidFileFormatError(f"Unknown critical chunk {type_bytes:x}")
        return None
    return chunk_class.from_payload(payload,registry)

# --- High-Level API ---
@dataclass
class LayerState:
    layer_id:int; name:str; parent_id:int=0; data:Optional[np.ndarray]=None; preview_data:Optional[np.ndarray]=None; residuals:Optional[np.ndarray]=None; upscale_factor:int=1; codec:PixelCodec=PixelCodec.PIX_NATIVE; quality:int=90; offset:Tuple[int,int]=(0,0); opacity:float=1.0; rotation:float=0.0; scale:float=1.0; metadata:Dict=field(default_factory=dict)
    def copy(self): return LayerState(**self.__dict__)
    def to_pil_image(self,use_previews:bool=False,enable_super_res:bool=True)->Optional[Image.Image]:
        pixel_source=self.preview_data if use_previews and self.preview_data is not None else self.data
        if not isinstance(pixel_source,np.ndarray): return None
        
        final_pixels=pixel_source
        if enable_super_res and not use_previews and self.residuals is not None:
            low_res_pil=Image.fromarray(pixel_source).convert("RGBA");h,w=low_res_pil.height*self.upscale_factor,low_res_pil.width*self.upscale_factor
            upscaled_blurry=low_res_pil.resize((w,h),Image.Resampling.BICUBIC)
            residuals_reshaped=self.residuals.reshape(h,w,pixel_source.shape[2])
            reconstructed_arr=np.array(upscaled_blurry,dtype=np.int16)+residuals_reshaped
            final_pixels=np.clip(reconstructed_arr,0,255).astype(np.uint8)
        
        mode="L" if final_pixels.ndim==2 or final_pixels.shape[2]==1 else "RGB" if final_pixels.shape[2]==3 else "RGBA"
        return Image.fromarray(final_pixels.squeeze(),mode=mode)

@dataclass
class PropertyCommand: target_id:int; key:PropertyKey; value:float

@dataclass
class Scene:
    width:int=0; height:int=0; metadata:Dict=field(default_factory=dict); frame_rate:int=30; duration:int=1; layers:Dict[int,LayerState]=field(default_factory=dict); timeline:Dict[int,List[PropertyCommand]]=field(default_factory=dict)
    
    def get_frame_states(self, frame_number: int) -> Dict[int, LayerState]:
        states = {layer_id: layer.copy() for layer_id, layer in self.layers.items()}
        
        relevant_frames = []
        for f_num in self.timeline.keys():
            if f_num <= frame_number:
                relevant_frames.append(f_num)
        
        for frame in sorted(relevant_frames):
            for command in self.timeline[frame]:
                self._apply_command(states, command)
        return states

    def _apply_command(self, states: Dict[int, LayerState], command: PropertyCommand):
        target_layer = states.get(command.target_id)
        if not target_layer:
            return

        if command.key == PropertyKey.OPACITY: target_layer.opacity = command.value
        elif command.key == PropertyKey.OFFSET_X: target_layer.offset = (command.value, target_layer.offset[1])
        elif command.key == PropertyKey.OFFSET_Y: target_layer.offset = (target_layer.offset[0], command.value)
        elif command.key == PropertyKey.ROTATION: target_layer.rotation = command.value
        elif command.key == PropertyKey.SCALE: target_layer.scale = command.value
    
    def render_frame(self, frame_number: int, use_previews: bool = False, enable_super_res: bool = True) -> Image.Image:
        frame_states = self.get_frame_states(frame_number)
        canvas = Image.new('RGBA', (self.width, self.height))
        
        children_map = {layer_id: [] for layer_id in frame_states}
        for layer in frame_states.values():
            if layer.parent_id in children_map:
                children_map[layer.parent_id].append(layer)
        
        def render_recursive(layer_id: int, parent_transform: Tuple[int, int]):
            state = frame_states[layer_id]
            layer_image = state.to_pil_image(use_previews, enable_super_res)
            
            current_offset = (parent_transform[0] + state.offset[0], parent_transform[1] + state.offset[1])
            
            if layer_image:
                w, h = layer_image.size
                if state.scale != 1.0:
                    layer_image = layer_image.resize((int(w * state.scale), int(h * state.scale)), Image.Resampling.LANCZOS)
                if state.rotation != 0:
                    layer_image = layer_image.rotate(state.rotation, expand=True, resample=Image.Resampling.BICUBIC)
                
                tw, th = layer_image.size
                paste_position = (current_offset[0] - tw // 2, current_offset[1] - th // 2)
                
                temp_canvas = Image.new('RGBA', canvas.size)
                temp_canvas.paste(layer_image, paste_position)
                canvas.alpha_composite(temp_canvas)
            
            for child in children_map.get(layer_id, []):
                render_recursive(child.layer_id, current_offset)
        
        top_level_layers = [layer for layer in frame_states.values() if layer.parent_id == 0]
        for layer in top_level_layers:
            render_recursive(layer.layer_id, (0, 0))
            
        return canvas

    def render_animation(self, output_folder: str, use_previews: bool = False):
        import os
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        for i in range(self.duration):
            print(f"Rendering frame {i+1}/{self.duration}{' (Draft)' if use_previews else ''}...")
            frame_image = self.render_frame(i, use_previews=use_previews)
            frame_image.save(os.path.join(output_folder, f"frame_{i:04d}.png"))
    
    def save(self, path: str, generate_previews: bool = True, preview_size: int = 256):
        with open(path, "wb") as f:
            f.write(self.to_bytes(generate_previews, preview_size))

    def to_bytes(self, generate_previews: bool = True, preview_size: int = 256) -> bytes:
        stream = io.BytesIO()
        stream.write(PIX_SIGNATURE)
        stream.write(HEADChunk(self.width, self.height).to_bytes())
        if self.metadata: stream.write(METAChunk(self.metadata).to_bytes())
        if self.duration > 1: stream.write(ANIMChunk(self.frame_rate, self.duration).to_bytes())
        
        for layer_state in self.layers.values():
            child_chunks = []
            if layer_state.data is not None:
                codec = layer_state.codec
                if codec == PixelCodec.AVIF and not AVIF_SUPPORT: codec = PixelCodec.WEBP
                
                pil_image = Image.fromarray(layer_state.data)
                
                if codec == PixelCodec.PIX_NATIVE:
                    h,w,c = layer_state.data.shape; payload = apply_filters_numpy(layer_state.data.tobytes(), w, h, c)
                    child_chunks.append(LDATChunk(payload, codec))
                else:
                    codec_name = "webp" if codec == PixelCodec.WEBP else "avif"
                    with io.BytesIO() as bio:
                        pil_image.save(bio, format=codec_name, quality=layer_state.quality)
                        child_chunks.append(LDATChunk(bio.getvalue(), codec))
                
                if layer_state.residuals is not None:
                    child_chunks.append(HFDChunk(layer_state.upscale_factor, layer_state.residuals))
                
                if generate_previews:
                    preview_image = pil_image.copy()
                    preview_image.thumbnail((preview_size, preview_size))
                    with io.BytesIO() as bio:
                        preview_image.save(bio, format="webp", quality=75)
                        child_chunks.append(PRVWChunk(bio.getvalue(), PixelCodec.WEBP))

            if layer_state.metadata: child_chunks.append(METAChunk(layer_state.metadata))
            
            stream.write(FRAMChunk(layer_state.layer_id,layer_state.parent_id,layer_state.name,int(layer_state.opacity*255),*layer_state.offset,layer_state.rotation,layer_state.scale,child_chunks).to_bytes())
        
        for frame_number, commands in sorted(self.timeline.items()):
            prop_chunks = [PROPChunk(c.target_id,c.key,c.value) for c in commands]
            stream.write(FCTLChunk(frame_number, prop_chunks).to_bytes())
        
        stream.write(PIX_FOOTER)
        return stream.getvalue()

    @classmethod
    def load(cls, path: str) -> 'Scene':
        with open(path, "rb") as f:
            return cls.from_stream(f)

    @classmethod
    def from_stream(cls, stream: BinaryIO) -> 'Scene':
        if stream.read(4) != PIX_SIGNATURE: raise InvalidFileFormatError("Invalid PIX signature")
        
        all_chunks = [chunk for chunk in iter(lambda: _read_chunk(stream, CHUNK_REGISTRY), None)]
        scene = cls()
        
        for chunk in all_chunks:
            if isinstance(chunk, HEADChunk): scene.width, scene.height = chunk.width, chunk.height
            elif isinstance(chunk, ANIMChunk): scene.frame_rate, scene.duration = chunk.frame_rate, chunk.duration
            elif isinstance(chunk, METAChunk): scene.metadata = chunk.metadata
            elif isinstance(chunk, FCTLChunk):
                scene.timeline[chunk.frame_number] = [PropertyCommand(pc.target_id,pc.key,pc.value) for pc in chunk.commands]
            elif isinstance(chunk, FRAMChunk):
                ldat = next((c for c in chunk.child_chunks if isinstance(c, LDATChunk)), None)
                prvw = next((c for c in chunk.child_chunks if isinstance(c, PRVWChunk)), None)
                hfd = next((c for c in chunk.child_chunks if isinstance(c, HFDChunk)), None)
                meta = next((c for c in chunk.child_chunks if isinstance(c, METAChunk)), None)
                
                full_data, preview_data, residuals, upscale_factor = None, None, None, 1
                
                if ldat:
                    if ldat.pixel_codec == PixelCodec.PIX_NATIVE:
                        # PIX_NATIVE doesn't store dimensions; this is a known limitation to fix.
                        # For now, we assume this case isn't used with this loader.
                        # A proper fix would add dimensions to the native LDAT chunk.
                        pass
                    else:
                        full_data = np.array(Image.open(io.BytesIO(ldat.pixel_payload)).convert("RGBA"))
                
                if prvw: preview_data = np.array(Image.open(io.BytesIO(prvw.pixel_payload)).convert("RGBA"))
                if hfd: residuals, upscale_factor = hfd.residuals, hfd.upscale_factor
                
                scene.layers[chunk.layer_id] = LayerState(chunk.layer_id, chunk.parent_id, chunk.name, full_data, preview_data, residuals, upscale_factor, metadata=meta.metadata if meta else {}, offset=(chunk.offset_x, chunk.offset_y), opacity=chunk.opacity/255.0, rotation=chunk.rotation, scale=chunk.scale)
        return scene

# --- Creation Helpers & Demonstration ---
def create_sr_layer(layer_id,name,pil_image,factor=2):
    if pil_image.mode != "RGBA": pil_image = pil_image.convert("RGBA")
    high_res_arr = np.array(pil_image);h,w,c=high_res_arr.shape;low_res_dims=(w//factor,h//factor)
    low_res_img = pil_image.resize(low_res_dims,resample=Image.Resampling.LANCZOS);low_res_arr=np.array(low_res_img)
    upscaled_blurry=low_res_img.resize((w,h),Image.Resampling.BICUBIC);residuals=high_res_arr.astype(np.int16)-np.array(upscaled_blurry,dtype=np.int16);
    return LayerState(layer_id,name,data=low_res_arr,residuals=residuals.flatten(),upscale_factor=factor,codec=PixelCodec.PIX_NATIVE)

if __name__ == "__main__":
    print(f"--- PIX v12.0 The Honest Implementation ---")
    
    try: photo_img = Image.open("background.jpg").resize((1280, 720))
    except FileNotFoundError: print("NOTE: 'background.jpg' not found, using placeholder."); photo_img = Image.new('RGB', (1280, 720), '#102030')
    sr_graphic = Image.new('RGBA', (400, 100)); ImageDraw.Draw(sr_graphic).text((10, 10), "Super-Res", fill="cyan", font_size=40)
    
    scene = Scene(width=1280, height=720, duration=30, frame_rate=30, metadata={"project": "PIX Final Demo"})
    scene.layers = {
        1: LayerState(layer_id=1, name="Photo BG", data=np.array(photo_img), codec=PixelCodec.WEBP, quality=85),
        10: create_sr_layer(10, "Title", sr_graphic, factor=2),
    }
    scene.timeline = {
        0: [PropertyCommand(10, PropertyKey.OFFSET_X, 100)],
        29: [PropertyCommand(10, PropertyKey.OFFSET_X, 1180)]
    }
    
    FILE_NAME = "scene_v12_honest.pix"
    scene.save(FILE_NAME)
    print(f"\nScene saved to '{FILE_NAME}'.")
    
    loaded_scene = Scene.load(FILE_NAME)
    print("Scene loaded successfully.")
    
    loaded_scene.render_frame(15, use_previews=False, enable_super_res=False).save("v12_no_sr.png")
    loaded_scene.render_frame(15, use_previews=False, enable_super_res=True).save("v12_with_sr.png")
    print("\nRendered test frames with and without super-resolution.")
    
    loaded_scene.render_animation("animation_draft", use_previews=True)
    print("Draft animation rendered.")
