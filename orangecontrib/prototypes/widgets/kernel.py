import io
import sys

import pickle
import traceback
from typing import Mapping, Any


from ipykernel.ipkernel import IPythonKernel


if sys.version_info < (3, 8):
    import pickle5 as pickle


class TransferMixin:
    def push_namespace_request(self: IPythonKernel, stream, ident, parent):
        """handle an push_namespace request

        The message content must be
        {
            'names': []  # list of (variable) names to update.
        }
        The contents must be supplied via the messages 'buffers' which is a
        where the first buffer is a pickle (format 5) pickle stream and the rest
        are the 'out-of-band' buffers for the pickle 5 protocol
        """
        try:
            content = parent['content']
            names = content["names"]
            buffers = parent["buffers"]
            assert all(isinstance(name, str) for name in names)
        except KeyError:
            self.log.error("Got bad msg: ")
            self.log.error("%s", parent)
            return

        metadata = self.init_metadata(parent)
        ns = {}
        try:
            head, *buffers = buffers
            unpickler = pickle.Unpickler(io.BytesIO(head), buffers=buffers)
            for name in names:
                name_ = unpickler.load()
                obj = unpickler.load()
                assert name == name_
                ns[name] = obj
            reply_content = {"status": "ok"}
        except Exception:
            reply_content = {"status": "error", "traceback": traceback.format_tb()}

        # Send the reply.
        metadata = self.finish_metadata(parent, metadata, reply_content)

        reply_msg = self.session.send(
            stream, 'push_namespace_reply', reply_content, parent,
            metadata=metadata, ident=ident
        )
        self.shell.push(ns)
        self.log.debug("%s", reply_msg)

    def get_namespace_request(self: IPythonKernel, stream, ident, parent):
        """handle an get_namespace request"""
        try:
            content = parent['content']
            names = content["names"]
            assert all(isinstance(name, str) for name in names)
        except KeyError:
            self.log.error("Got bad msg: ")
            self.log.error("%s", parent)
            return
        metadata = self.init_metadata(parent)

        ns = {name: self.shell.user_ns.get(name) for name in names}
        pickle_buffer = io.BytesIO()
        buffers = []
        pickler = pickle.Pickler(
            pickle_buffer, pickle.HIGHEST_PROTOCOL,
            buffer_callback=buffers.append)
        try:
            for name in names:
                pickler.dump((name, ns[name]))
        except Exception:
            buffers = None
            reply_content = {"status": "error", "traceback": traceback.format_tb()}
        else:
            buffers = [pickle_buffer.getbuffer(), *[pb.raw() for pb in buffers]]
            reply_content = {"status": "ok", "names": names}

        # Send the reply.
        metadata = self.finish_metadata(parent, metadata, reply_content)
        reply_msg = self.session.send(
            stream, 'get_namespace_reply', reply_content, parent,
            metadata=metadata, ident=ident, buffers=buffers
        )
        self.log.debug("%s", reply_msg)


class IpyKernelPushGet(IPythonKernel, TransferMixin):
    """
    An IPythonKernel implementing 'push/get_namespace_requests' for setting
    up and extracting shell user namespace.
    """
    msg_types = IPythonKernel.msg_types + [
        "push_namespace_request",
        "get_namespace_request",
    ]

