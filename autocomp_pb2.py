# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: autocomp.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0e\x61utocomp.proto\x12\x0c\x61utocomplete\"!\n\x10InferenceRequest\x12\r\n\x05input\x18\x01 \x01(\t\"#\n\x11InferenceResponse\x12\x0e\n\x06output\x18\x01 \x01(\t2\\\n\x0c\x41utoComplete\x12L\n\tFetchNext\x12\x1e.autocomplete.InferenceRequest\x1a\x1f.autocomplete.InferenceResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'autocomp_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_INFERENCEREQUEST']._serialized_start=32
  _globals['_INFERENCEREQUEST']._serialized_end=65
  _globals['_INFERENCERESPONSE']._serialized_start=67
  _globals['_INFERENCERESPONSE']._serialized_end=102
  _globals['_AUTOCOMPLETE']._serialized_start=104
  _globals['_AUTOCOMPLETE']._serialized_end=196
# @@protoc_insertion_point(module_scope)
