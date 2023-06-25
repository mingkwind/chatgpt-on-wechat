import asyncio, json
import plugins.plugin_bingchat.binggpt.myEdgeGPT as myEdgeGPT
import os
import uuid
import time
import re

STYLES = ["balanced", "creative", "precise"]
# 一个容量为100的字典，用于存储聊天机器人
Capacity = 100
CHATBOTS = {}
def add_bot(token, chatbot):
    while len(CHATBOTS) >= Capacity:
        # 按照时间戳排序，删除最早的一个
        oldest = min(CHATBOTS, key=lambda k: CHATBOTS[k]["useTimeStamp"])
        del CHATBOTS[oldest]
    CHATBOTS[token] = {}
    CHATBOTS[token]["chatBot"] = chatbot

curdir = os.path.dirname(__file__)
COOKIE_FILE_PATH = os.path.join(curdir, "cookies.json")
BASE_URL = "https://sydney.bing.com/"
WSS_LINK = "wss://sydney.bing.com/sydney/ChatHub"
endpoint_url = json.loads(open(os.path.join(curdir,"config.json"), encoding="utf-8").read()).get("bing_base_url")
if endpoint_url:
    BASE_URL = endpoint_url
    WSS_LINK = BASE_URL.replace("https://", "wss://") + "sydney/ChatHub"
ALL_PROXY = os.environ.get("all_proxy")


def getTimeStamp() -> str:
    return int(time.time())


def getChatBot(token: str) -> tuple:
    global CHATBOTS
    if token:
        if token in CHATBOTS:
            chatBot = CHATBOTS.get(token).get("chatBot")
        else:
            cookies = json.loads(open(COOKIE_FILE_PATH, encoding="utf-8").read())
            chatBot = myEdgeGPT.Chatbot(cookies=cookies, base_url=BASE_URL, proxy=ALL_PROXY)
            add_bot(token, chatBot)
    else:
        return token, None
    CHATBOTS[token]["useTimeStamp"] = getTimeStamp()
    return token, chatBot


def getStyleEnum(style: str) -> myEdgeGPT.ConversationStyle:
    enum = myEdgeGPT.ConversationStyle
    if style == "balanced":
        enum = enum.balanced
    elif style == "creative":
        enum = enum.creative
    elif style == "precise":
        enum = enum.precise
    return enum


def filterAnswer(answer: str) -> str:
    answer = re.sub(r"\[\^.*?\^]", "", answer)
    # 把** **包裹的内容，去掉两个**，只保留内容
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    return answer


def getAnswer(data: dict) -> str:
    messages = data.get("item").get("messages")
    if "text" in messages[1]:
        return messages[1].get("text")
    else:
        return messages[1].get("adaptiveCards")[0].get("body")[0].get("text")


def getStreamAnswer(data: dict) -> str:
    messages = data.get("item").get("messages")
    if "text" in messages[1]:
        answer = messages[1].get("text")
    else:
        answer = messages[1].get("adaptiveCards")[0].get("body")[0].get("text")
    answer = filterAnswer(answer)
    return answer


def needReset(data: dict, answer: str) -> bool:
    maxTimes = (
        data.get("item").get("throttling").get("maxNumUserMessagesInConversation")
    )
    nowTimes = data.get("item").get("throttling").get("numUserMessagesInConversation")
    errorAnswers = ["I’m still learning", "我还在学习"]
    if [errorAnswer for errorAnswer in errorAnswers if errorAnswer in answer]:
        return True
    elif nowTimes == maxTimes:
        return True
    return False


def getUrl(data: dict) -> list:
    sourceAttributions = data.get("item").get("messages")[1].get("sourceAttributions")
    urls = []
    if sourceAttributions:
        for sourceAttribution in sourceAttributions:
            urls.append(
                {
                    "title": sourceAttribution.get("providerDisplayName"),
                    "url": sourceAttribution.get("seeMoreUrl"),
                }
            )
    return urls


async def chat(token, question, style):
    if not style or not question:
        return "参数错误"
    elif style not in STYLES:
        return "style参数错误"

    token, chatBot = getChatBot(token)
    if not chatBot:
        return "token不存在"
    data = await chatBot.ask(question, conversation_style=getStyleEnum(style), wss_link = WSS_LINK)

    if data.get("item").get("result").get("value") == "Throttled":
        return "已上限,24小时后尝试"

    info = {"answer": "", "urls": [], "reset": False, "token": token}
    answer = getAnswer(data)
    answer = filterAnswer(answer)
    info["answer"] = answer
    info["urls"] = getUrl(data)

    if needReset(data, answer):
        await chatBot.reset()
        info["reset"] = True

    return info

def GetReply(token,question,style):
    data = asyncio.run(chat(token,question, style))
    if isinstance(data, str):
        return data, None
    answer = data.get("answer")
    urls = data.get("urls")
    if urls:
        answer += "\n" + "[" + "\n".join([url.get("title") + ": " + url.get("url") for url in urls]) + "]"
    if data.get("reset"):
        answer += "\n" + "已重置"
    return answer

def ResetBot(token):
    # 重置聊天机器人
    # 删除token对应的聊天机器人
    global CHATBOTS
    if token in CHATBOTS:
        del CHATBOTS[token]
    return "已重置"

if __name__ == "__main__":
    token = str(uuid.uuid4())
    question = "你好"
    style = "balanced"
    answer = GetReply(token, question, style)
    print(answer)
    question = "鸡你太美"
    answer = GetReply(token, question, style)
    print(answer)