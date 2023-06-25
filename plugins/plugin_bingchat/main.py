from chatgpt_tool_hub.apps import AppFactory
from chatgpt_tool_hub.apps.app import App
from chatgpt_tool_hub.tools.all_tool_list import get_all_tool_names

import plugins
from bridge.bridge import Bridge
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common import const
from common.log import logger
from config import conf
from plugins import *
import plugins.plugin_bingchat.binggpt.GetReply as GetReply

@plugins.register(
    name="bingchat",
    desc="Use NewBing to have an Internet-enabled chat!",
    version="0.1",
    author="mingkwind",
    desire_priority=0,
)
class BingChat(Plugin):
    def __init__(self):
        super().__init__()
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context

    def get_help_text(self, verbose=False, **kwargs):
        help_text = "可以使用newbing进行联网查询和聊天。style=0/1/2分别对应precise/balanced/creative。"
        return help_text

    def on_handle_context(self, e_context: EventContext):
        if e_context["context"].type != ContextType.TEXT:
            return

        # 暂时不支持未来扩展的bot
        if Bridge().get_bot_type("chat") not in (
            const.CHATGPT,
            const.OPEN_AI,
            const.CHATGPTONAZURE,
        ):
            return
        # 获取微信uid   
        user_id = e_context["context"]["session_id"]


        content = e_context["context"].content
        content_list = e_context["context"].content.split(maxsplit=1)

        if not content or len(content_list) < 1:
            e_context.action = EventAction.CONTINUE
            return

        logger.info("[bing] on_handle_context. content: %s" % content)
        reply = Reply()
        reply.type = ReplyType.TEXT
        trigger_prefix = conf().get("plugin_trigger_prefix", "$")
        if content.startswith(f"{trigger_prefix}bing"):
            if len(content_list) == 1:
                logger.debug("[bing]: get help")
                reply.content = self.get_help_text()
            elif len(content_list) > 1:
                # 判断是否开头有style=0/1/2
                # 分别对应precise/balanced/creative
                stylemap = {"0": "precise", "1": "balanced", "2": "creative"}
                style = "balanced"
                # 格式为 #bing style=0/1/2 xxx
                if content_list[1].startswith("style="):
                    styleId = content_list[1].split(maxsplit=1)[0].split("=")[1]
                    if styleId in stylemap:
                        style = stylemap[styleId]
                    else:
                        logger.debug("[bing]: get help")
                        reply.content = self.get_help_text()
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    content_list[1] = content_list[1].split(maxsplit=1)[1]
                if len(content_list) == 1:
                    logger.debug("[bing]: get help")
                    reply.content = self.get_help_text()
                else:
                    query = content_list[1].strip()
                    all_sessions = Bridge().get_bot("chat").sessions
                    user_session = all_sessions.session_query(query, e_context["context"]["session_id"]).messages
                    print(e_context["context"]["session_id"])
                    # for循环最多3次，防止死循环
                    have_reply = False
                    for i in range(3):
                        try:
                            _reply = GetReply.GetReply(e_context["context"]["session_id"],content_list[1], style)
                            e_context.action = EventAction.BREAK_PASS
                            all_sessions.session_reply(_reply, e_context["context"]["session_id"])
                            have_reply = True
                            break
                        except Exception as e:
                            logger.exception(e)
                            logger.error(str(e))
                            GetReply.ResetBot(e_context["context"]["session_id"])
                    if not have_reply:
                        logger.error("BingChat: get reply 3 times failed")
                        e_context["context"].content = "出错惹，换个词试试吧！"
                        reply.type = ReplyType.ERROR
                        e_context.action = EventAction.BREAK
                        return
                reply.content = _reply
                e_context["reply"] = reply
        return