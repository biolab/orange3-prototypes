import re

from concurrent.futures import Future
from typing import Mapping, Any, List, Callable, Dict

from qtconsole.client import QtKernelClient

from .kernel_utils import dump, load


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
    Push the namespace (`ns`) in to the remote kernel's interactive shell.

    The remote kernel must respond to 'push_namespace_request' messages.

    Return a future with the response message.
    """
    names = list(ns.keys())
    buffers = dump(ns)
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


def get_namespace(
        client: QtKernelClient, names: List[str]
) -> 'Future[Dict[str, Any]]':
    """
    Extract named variables (`names`) from the remote kernel shell namespace.

    The remote kernel must respond to 'get_namespace_request' messages.

    Return a Future with the result namespace.
    """
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
            try:
                ns = load(buffers)
                assert isinstance(ns, dict)
                assert set(ns.keys()) == set(names)
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(ns)
        else:
            f.set_exception(Exception(str(msg)))
    return f


def remove_ansi_control(content: str) -> str:
    """Remove ansi terminal control sequences from `content`."""
    return re.sub("\x1b[\\[0-9;]+m", "", content)
