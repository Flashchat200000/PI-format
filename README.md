
Version: 4.2
Status: Stable

1. Overview

The PI Ultimate (.pixu) format is a container for 3D scene data, designed for robustness, extensibility, and high-performance, asynchronous loading. It utilizes a declarative task graph to define rendering logic and dependencies.

The PI Secure (.pixs) format is a cryptographic wrapper around a standard .pixu file, providing end-to-end encryption for secure asset distribution.

2. General Principles
2.1. Endianness

All multi-byte integer types (uint16_t, uint32_t, uint64_t, int64_t) are stored in Big-Endian (Network Byte Order).

2.2. String Encoding

Strings are stored with a uint32_t length prefix, followed by the character data encoded in UTF-8.

Type	Name	Description
uint32_t	Length	Number of bytes in the string.
char[Length]	Data	UTF-8 encoded string data.
2.3. Data Integrity

Data integrity is ensured via CRC32 checksums for critical data sections.

3. PI Ultimate (.pixu) File Structure

A .pixu file is composed of three main parts: a fixed-size header, a variable-size collection of data segments, and a variable-size Master Block (index).

Component	Description
File Header	Fixed-size block containing metadata about the file.
Data Segments	Unordered collection of compressed resource data chunks.
Master Block	The central index describing all content in the file.
Master Block Checksum	A CRC32 checksum of the entire Master Block.
3.1. File Header

The first 22 bytes of the file.

Offset (bytes)	Size (bytes)	Type	Description
0	4	uint32_t	Signature: 0x50495855 ("PIXU")
4	2	uint16_t	Specification version (e.g., 4).
6	8	uint64_t	Feature Flags (bitmask). See Section 3.2.1.
14	8	uint64_t	Byte offset from the start of the file to the Master Block.
3.2. Master Block

The Master Block is a stream of typed, size-prefixed data blocks. This structure allows new block types to be added in future versions while maintaining backward compatibility.

A parser reads the Block ID, reads the Block Size, and then can either parse the block's data or skip Block Size bytes to get to the next block.

Each block has the following structure:

Type	Name	Description
uint8_t	Block ID	An enum value identifying the block type.
uint64_t	Block Size	The total size in bytes of the Block Data field.
byte[Size]	Block Data	The payload of the block.
3.2.1. Block Types (Block ID)
ID	Constant Name	Description
1	kMetadata	Contains key-value string metadata.
2	kResources	A list of all addressable resources.
3	kFallbacks	A list of fallback data items.
4	kTaskGraph	The declarative task graph for rendering.
3.2.2. Block: Metadata (ID = 1)

A simple dictionary of key-value pairs.

Type	Name	Description
uint64_t	Pair Count	Number of key-value pairs.
...	Pairs	Pair Count instances of the following:
string	Key	The metadata key.
string	Value	The metadata value.
3.2.3. Block: Resources (ID = 2)

Defines all primary data assets.

Type	Name	Description
uint64_t	Res. Count	Number of ResourceDescriptor entries.
...	Descriptors	Res. Count instances of ResourceDescriptor.

ResourceDescriptor Structure:

Type	Name	Description
uint64_t	ID	Unique identifier for this resource.
uint16_t	Type	Enum (ResourceType) describing the data.
string	Name	Human-readable name (e.g., "player.mesh").
uint64_t	Seg. Count	Number of data segments for this resource.
DataSegmentLocation[Count]	Segments	An array of DataSegmentLocation structs.

DataSegmentLocation Structure:

Type	Name	Description
uint64_t	Offset	Byte offset to the compressed data segment.
uint32_t	Compressed Size	Size of the compressed data on disk.
uint32_t	Uncompressed Size	Size of the data after decompression.
uint32_t	CRC32 Checksum	Checksum of the compressed data.
3.2.4. Block: Task Graph (ID = 4)

Defines the directed acyclic graph (DAG) of operations.

Type	Name	Description
uint64_t	Node Count	Number of GraphNode entries.
...	Nodes	Node Count instances of GraphNode.
uint64_t	Root Node ID	The ID of the final node in the graph.

GraphNode Structure:

Type	Name	Description
uint64_t	ID	Unique identifier for this node.
uint16_t	Type	Enum (NodeType) of the operation.
string	Intent	Human-readable description of the node.
uint64_t	Input Count	Number of node IDs this node depends on.
uint64_t[Count]	Inputs	An array of input GraphNode IDs.
uint64_t	Param Count	Number of parameters for this node.
...	Params	Param Count instances of key-value parameters.

NodeParameter Structure: This is a variant type.

Type	Name	Description
uint8_t	Type Index	0: monostate, 1: int64, 2: string.
...	Value	The value, with type matching the index.
3.3. Master Block Checksum

The final 4 bytes of the file are a uint32_t CRC32 checksum calculated over the entire Master Block (from its start at Master Block Offset to the end of the file, minus these 4 bytes).

4. PI Secure (.pixs) File Structure

The .pixs file is a wrapper that contains a fully-formed, encrypted .pixu file.

Component	Description
Secure Header	Contains format info and encrypted session keys.
Encrypted Payload	An entire .pixu file, encrypted symmetrically.
4.1. Secure Header
Offset (bytes)	Size (bytes)	Type	Description
0	4	uint32_t	Signature: 0x50495853 ("PIXS")
4	2	uint16_t	Specification version (e.g., 4).
6	8	uint64_t	Number of recipients.
14	...	...	Recipient Count instances of Recipient Blocks.

Recipient Block Structure:

Type	Name	Description
string	Recipient ID	Identifier for the key owner (e.g., "bob").
uint32_t	Enc. Key Length	Size of the encrypted session key in bytes.
byte[Length]	Encrypted Session Key	The session key, asymmetrically encrypted with the recipient's public key.
4.2. Encrypted Payload

This section immediately follows the last Recipient Block. It contains the complete binary data of a .pixu file (including its header, data segments, and master block), symmetrically encrypted using the decrypted session key
