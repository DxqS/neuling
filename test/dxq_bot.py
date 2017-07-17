# coding:utf-8
'''
Created on 2017/6/27.

@author: chk01
'''
import datetime
import logging
import re
import time

import psutil

from wxpy import *
from wxpy.utils import start_new_thread

logging.basicConfig(level=logging.WARNING)

bot = Bot(cache_path=True, console_qr=True)
admin = ensure_one(bot.friends().search(remark_name="孙大款"))
admins = bot.self
group_test = bot.groups().search("数为研发部")[0]
# group_test = bot.groups().search("bot")[0]
xiaoi = XiaoI(key='xv14dA0oeES7', secret='xMZavtxt0TEx5G6qyWqe')
welcome_text = '''🎉欢迎《{}》加入数为研发部！！！🙈🙈🙈
📖 研发网盘地址 //192.168.1.21 
😃 账号密码swyanfa yanfa123
'''
# 新人入群通知的匹配正则
rp_new_member_name = (
    re.compile(r'^"(.+)"通过'),
    re.compile(r'邀请"(.+)"加入'),
)
daily_report_list = {
    "9": "天天好心情！",
    "11": "午饭",
    "18": "正常下班",
    "20": "餐补下班",
    "22": "交补下班",
    "23": "该睡觉了",
    "Sunday": "周报！周报！周报！",
    "nine": "工资即将到账"
}


# @提醒
@bot.register(group_test, TEXT)
def at_warn(msg):
    if msg.is_at:
        msg.member.send("听说你@了我，我可能在忙没看见，最好去吼我下谢谢！")


def get_new_member_name(msg):
    # itchat 1.2.32 版本未格式化群中的 Note 消息
    from itchat.utils import msg_formatter
    msg_formatter(msg.raw, 'Text')

    for rp in rp_new_member_name:
        match = rp.search(msg.text)
        if match:
            return match.group(1)


# 新人欢迎消息
@bot.register(group_test, NOTE)
def welcome(msg):
    name = get_new_member_name(msg)
    if name:
        return welcome_text.format(name)


#######报告心跳##########
def send_iter(receiver, iterable):
    """
    用迭代的方式发送多条消息
    :param receiver: 接收者
    :param iterable: 可迭代对象
    """

    if isinstance(iterable, str):
        raise TypeError

    for msg in iterable:
        receiver.send(msg)


process = psutil.Process()


def status_text():
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())
    memory_usage = process.memory_info().rss
    yield '[now] {now:%H:%M:%S}\n[uptime] {uptime}\n[memory] {memory}\n[messages] {messages}'.format(
        now=datetime.datetime.now(),
        uptime=str(uptime).split('.')[0],
        memory='{:.2f} MB'.format(memory_usage / 1024 ** 2),
        messages=len(bot.messages)
    )


# 定时报告进程状态
def heartbeat():
    while bot.alive:
        time.sleep(600)
        # noinspection PyBroadException
        try:
            # logger.warning(status_text())
            send_iter(admin, status_text())
        except:
            pass
            # logger.exception('failed to report heartbeat:')


# 休息时间勿扰
@bot.register(Friend, TEXT)
def auto_reply(msg):
    HOUR = int(time.strftime("%H", time.localtime(int(time.time()))))
    if msg.sender != admin:
        if HOUR < 9 or HOUR >= 22:
            msg.sender.send("您好，主人还在休息。如有急事，请电联18768120187")


def daily_report():
    ts = int(time.time())
    _TIME = list(map(int, time.strftime('%Y-%m-%d-%H-%M', time.localtime(ts)).split("-")))
    weekday = int(datetime.datetime.now().weekday())
    report = ''
    if weekday == 6:
        report = daily_report_list['Sunday']
    else:
        if _TIME[3] in [9, 11, 18, 20, 22, 23]:
            report = daily_report_list[str(_TIME[3])]
        else:
            if _TIME[2] == 9:
                report = daily_report_list['nine']
    return report


def report_daily():
    while bot.alive:
        time.sleep(1800)
        try:
            if daily_report():
                admin.send(daily_report())
        except:
            pass


###########远程操作##############
# 远程命令 (单独发给机器人的消息)


start_new_thread(heartbeat)
start_new_thread(report_daily)
bot.join()
