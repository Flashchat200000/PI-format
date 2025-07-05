
PIX is a high-performance, security-first file format designed for the next generation of web graphics. It combines procedural generation, cryptographic integrity, and a unique dual-mode architecture for universal compatibility.
PIX is not just another image format. It's a container that describes how to generate content, rather than just storing static pixels. This allows for incredibly rich, dynamic, and interactive graphics to be delivered in a minimal footprint, without compromising on security or performance.

Key Features

🔐 Cryptography-First Security: Every PIX file is protected by a mandatory cryptographic signature (e.g., ECDSA, RSA - specific provider is pluggable). The signature of the compressed data block is verified before any decompression or parsing occurs, mitigating a wide range of attacks. This is not CRC32; this is enterprise-grade integrity and authenticity.

🧠 Procedural Rendering: At its core, PIX stores a graph of procedural nodes, not just pixels. This means complex textures, gradients, and shapes can be described as a recipe in a few kilobytes, enabling infinite resolution and dynamic content.

🔄 Smart/Dumb Client Architecture: PIX solves the adoption problem with a built-in fallback cache. A "dumb" client (like a standard <img> tag) can display a simple preview, while a "smart" client (with the WASM renderer) can unlock the full, dynamic, procedural content.

🚀 Performance via WebAssembly: The core parsing and rendering logic is written in modern C++ for maximum performance and memory safety. It's compiled to a lightweight WebAssembly module, allowing it to run securely and efficiently in any modern browser.

Architectural Overview

PIX is designed for robustness and efficiency. The file structure and parsing logic are built to be secure and fast.

File Structure

The file is structured to allow for immediate verification and lazy loading. The footer is read first to locate critical sections.



Secure Loading & Parsing Logic

The SceneReader follows a strict, security-conscious procedure:

 code
1. Read Footer -> Get offsets for Data Block and Signature.
2. Read Header -> Get metadata (e.g., compression type).
3. Read Compressed Data Block into memory.
4. Read Signature Chunk.
5. --> VERIFY SIGNATURE of the compressed block. <-- CRITICAL STEP
6. If verification passes:
7.    Decompress data block into an in-memory stream.
8.    Parse the master object index.
9.    Lazily parse individual objects from the stream on demand.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Technical Specifications

Signature: PIX3 (v26)

Core Chunks: TREE (Scene Tree), NODE (Procedural Graph), MESH (3D Data), SCPT (Scripts), INDX (Master Index), SIGN (Signature).

Data Integrity: Cryptographic signature verification via a pluggable ICryptoProvider.

Architecture: Append-only friendly structure with a master index for fast object lookups.

Compatibility: Dual-mode access via fallback cache and full procedural rendering.

Getting Started

To build the reference implementation from the source, you will need:

A C++20 compatible compiler (GCC, Clang, MSVC)

CMake 3.15+

Generated bash
# Clone the repository
git clone https://github.com/Flashchat200000/PI-format.git
cd
https://github.com/Flashchat200000/PI-format.git

# Configure and build the project
cmake -B build
cmake --build build
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END


Roadmap

Expand the library of procedural nodes (noise, filters, etc.).

Implement additional compression providers (e.g., LZ4).

Enhance the animation system with physics-based curves.

Formalize the specification document.

License

This project is licensed under the MIT License. See the LICENSE file for details.
