import sys
import io
import pickle

from concurrent.futures import Future
from typing import Mapping, Any, List, Callable

from qtconsole.client import QtKernelClient

if sys.version_info < (3, 8):
    import pickle5 as pickle


HandlerType = Callable[[dict], Any]


def msg_response_observer(
        client: QtKernelClient, msg_type, msg_id, handler: HandlerType,
        *, channel="shell_channel",
) -> Callable[[], None]:
    def msg_recv(msg: dict):
        if msg['header']['msg_type'] == msg_type:
            if msg["parent_header"]["msg_id"] == msg_id:
                handler(msg)

    channel = getattr(client, channel)
    channel.message_received.connect(msg_recv)

    def disconnect():
        channel.message_received.disconnect(msg_recv)
    return disconnect


def on_reply(
        client: QtKernelClient, message_type: str, msg_id: str
) -> Callable[[HandlerType], None]:
    def wrapper(f: HandlerType) -> None:
        def f_(*args, **kwargs):
            disconnect()
            return f(*args, **kwargs)
        disconnect = msg_response_observer(client, message_type, msg_id, f_)
        f.disconnect = disconnect
    return wrapper


def on_message(
    client: QtKernelClient, message_type: str, msg_id: str,
    *, channel="shell_channel"
) -> Callable[[HandlerType], HandlerType]:
    def wrapper(f: HandlerType) -> HandlerType:
        disconnect = msg_response_observer(
            client, message_type, msg_id, f, channel=channel)
        f.disconnect = disconnect
        return f
    return wrapper


def push_namespace(
        client: QtKernelClient, ns: Mapping[str, Any]
) -> 'Future[dict]':
    """
    Push the namespace in to the remote kernels interactive shell.
    """
    buffer = io.BytesIO()
    buffers = []  # type: List[pickle.PickleBuffer]
    pickler = pickle.Pickler(buffer, protocol=5, buffer_callback=buffers.append)
    names = list(ns.keys())

    for name in names:
        pickler.dump(name)
        pickler.dump(ns[name])

    buffers = [buffer.getbuffer(), *[pb.raw() for pb in buffers]]
    content = dict(names=names)
    msg = client.session.msg('push_namespace_request', content)
    client.session.send(
        client.shell_channel.stream, msg, buffers=buffers
    )
    msg_id = msg['header']['msg_id']
    future = Future()

    @on_reply(client, "push_namespace_reply", msg_id)
    def _(msg):
        if not future.set_running_or_notify_cancel():
            return
        client.log.debug("Got: %s", msg)
        content = msg["content"]
        status = content["status"]
        if status == "ok":
            future.set_result(content)
        else:
            future.set_exception(Exception(content["status"]))
    return future


def get_namespace(client: QtKernelClient, names: List[str]) -> 'Future[dict]':
    content = dict(
        names=names,
    )
    msg = client.session.msg('get_namespace_request', content)
    client.shell_channel.send(msg)
    msg_id = msg['header']['msg_id']
    f = Future()

    @on_reply(client, "get_namespace_reply", msg_id)
    def _(msg):
        if not f.set_running_or_notify_cancel():
            return
        client.log.debug("Got: %s", msg)
        content = msg["content"]
        status = content["status"]

        if status == "ok":
            names = content["names"]
            buffers = msg["buffers"]
            head, *buffers = buffers
            unpickler = pickle.Unpickler(io.BytesIO(head), buffers=buffers)
            ns = {}
            try:
                for name in names:
                    name_, obj = unpickler.load()
                    assert name == name_
                    ns[name] = obj
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(ns)
        else:
            f.set_exception(Exception(str(msg)))
    return f
