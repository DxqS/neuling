#!/usr/bin/env python3
# coding: utf-8

"""
wxpy 机器人正在使用的代码

** 这些代码无法独立运行，但可用于参考 **

需要安装新内核分支 new-core
pip3 install -U git+https://github.com/youfou/wxpy.git@new-core

另外，还需要 psutil 模块，用于监控进程状态，例如内存占用情况。请自行安装:
pip3 install -U psutil
"""

import datetime
import os
import re
import subprocess
import time
from collections import Counter
from functools import wraps
from pprint import pformat

import psutil
from wxpy import *
from wxpy.utils import ensure_list, start_new_thread

from kick_votes import KickVotes
from remote import run_flask_app
from sms import send_sms
from timed_list import TimedList

# ---------------- 配置开始 ----------------

# Bot 对象初始化时的 console_qr 参数值
console_qr = True

# 机器人昵称 (防止登错账号)
bot_name = 'wxpy 机器人'

# 入群口令
group_code = 'wxpy'

# 自动回答关键词
kw_replies = {
    'wxpy 项目主页:\ngithub.com/youfou/wxpy': (
        '项目', '主页', '官网', '网站', 'github', '地址', 'repo', '版本'
    ),
    'wxpy 在线文档:\nwxpy.readthedocs.io': (
        '请问', '文档', '帮助', '怎么', '如何', '请教', '安装', '说明', '运行'
    ),
    '必看: 常见问题 FAQ:\nwxpy.readthedocs.io/faq.html': (
        'faq', '常见', '问题', '问答', '什么'
    ),
    (__file__, FILE): (
        '源码', '代码'
    )
}

# 新人入群的欢迎语
welcome_text = '''🎉 欢迎 @{} 的加入！
😃 请勿在本群使用机器人
📖 提问前请看 t.cn/R6VkJDy'''

help_info = '''😃 讨论主题
· 本群主题为 wxpy 与 Python
· 不限制其他话题，请区分优先级
· 支持分享对群员有价值的信息

⚠️ 注意事项
· 除群主外，勿在群内使用机器人
· 严禁灰产/黑产相关内容话题
· 请勿发布对群员无价值的广告

👮 投票移出
· 移出后将被拉黑 24 小时
· 请在了解事因后谨慎投票
· 命令格式: "移出 @人员"

🔧 实用链接
· 文档: url.cn/4660Oil
· 示例: url.cn/49t5O4x
· 项目: url.cn/463SJb8
'''

# ---------------- 配置结束 ----------------


logging.basicConfig(level=logging.DEBUG)

qr_path = 'static/qrcode.png'

sms_sent = False


def show_qrcode(core):
    global sms_sent

    Core.show_qrcode(core)

    if not sms_sent:
        # 发送短信
        send_sms()
        sms_sent = True


def logged_in(core):
    Core.remove_qrcode(core)


# noinspection PyUnusedLocal
def new_member(core, member):
    if member.group in groups:
        member.group.send(welcome_text.format(member.name))


# noinspection PyUnusedLocal
def deleting_member(core, member):
    admin_group.send('[member left]\nmember: {0.name}\ngroup: {1.name}'.format(member, member.group))


def _restart():
    os.execv(sys.executable, [sys.executable] + sys.argv)


def logged_out(core, reason):
    logger.critical('{} logged out:\n{}'.format(core, reason))
    _restart()


process = psutil.Process()


def _status_text():
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())
    memory_usage = process.memory_info().rss

    if globals().get('bot'):
        messages = bot.messages
    else:
        messages = list()

    return '[now] {now:%H:%M:%S}\n[uptime] {uptime}\n[memory] {memory}\n[messages] {messages}'.format(
        now=datetime.datetime.now(),
        uptime=str(uptime).split('.')[0],
        memory='{:.2f} MB'.format(memory_usage / 1024 ** 2),
        messages=len(messages)
    )


start_new_thread(run_flask_app, (qr_path, _status_text))

bot = Bot('bot.pkl', console_qr=True, qr_path=qr_path, hooks=dict(
    show_qrcode=show_qrcode,
    logged_in=logged_in,
    logged_out=logged_out,
    new_member=new_member,
    deleting_member=deleting_member,
))

if bot.name != bot_name:
    logging.error('Wrong User!')
    bot.logout()
    _restart()

admins = bot.friends.get(remark_name='Youfou'), bot.friends.get(remark_name='old-bot'), bot.self
admin_group = bot.groups.get('wxpy admins')
groups = list(map(lambda x: bot.groups.get(x), (
    'wxpy 交流群 🐰',
    'wxpy 交流群 🐱',
    'wxpy 交流群 🐨',
    'wxpy 交流群 🐹',
    'wxpy 交流群 🐿',
)))

test_group = bot.groups.get('wxpy test')
preview_group = bot.groups.get('wxpy preview')

# 初始化图灵机器人
tuling = Tuling()

# 远程踢人命令: 移出 @<需要被移出的人>
rp_kick = re.compile(r'^(?:移出|移除|踢出)\s*@(.+?)(?:\u2005?\s*$)')
kick_votes = KickVotes(300)
votes_to_kick = 5
black_list = TimedList()


def from_admin(msg):
    """
    判断 msg 的发送者是否为管理员
    """
    from_user = msg.member if isinstance(msg.chat, Group) else msg.sender
    return from_user in admins


def admin_auth(func):
    """
    装饰器: 验证函数的第 1 个参数 msg 是否来自 admins
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        msg = args[0]

        if from_admin(msg):
            return func(*args, **kwargs)
        else:
            raise ValueError('{} is not an admin!'.format(msg))

    return wrapped


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


def update_groups():
    yield 'updating groups...'
    for _group in groups:
        _group.update()
        yield '{}: {}'.format(_group.name, len(_group))


def status_text():
    yield _status_text()


# 定时报告进程状态
def heartbeat():
    while bot.alive:
        time.sleep(600)
        # noinspection PyBroadException
        try:
            send_iter(admin_group, status_text())
        except:
            logger.exception('failed to report heartbeat:\n')


start_new_thread(heartbeat)


def remote_eval(source):
    try:
        ret = eval(source, globals())
    except (SyntaxError, NameError):
        raise ValueError('got SyntaxError or NameError in source')

    logger.info('remote eval executed:\n{}'.format(source))
    yield pformat(ret)


def remote_shell(command):
    logger.info('executing remote shell cmd:\n{}'.format(command))
    r = subprocess.run(
        command, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    if r.stdout:
        yield r.stdout
    else:
        yield '[OK]'


def restart():
    yield 'restarting bot...'
    bot.core.dump()
    _restart()


def latency():
    yield '{:.2f}'.format(bot.messages[-1].latency)


# 远程命令 (单独发给机器人的消息)
remote_orders = {
    'g': update_groups,
    's': status_text,
    'r': restart,
    'l': latency,
}


@admin_auth
def server_mgmt(msg):
    """
    服务器管理:

        若消息文本为为远程命令，则执行对应函数
        若消息文本以 ! 开头，则作为 shell 命令执行
        若不满足以上，则尝试直接将 msg.text 作为 Python 代码执行
    """
    order = remote_orders.get(msg.text.strip())
    if order:
        logger.info('executing remote order: {}'.format(order.__name__))
        send_iter(msg.chat, order())
    elif msg.text.startswith('!'):
        command = msg.text[1:]
        send_iter(msg.chat, remote_shell(command))
    else:
        send_iter(msg.chat, remote_eval(msg.text))


def reply_by_keyword(msg):
    for reply, keywords in kw_replies.items():
        for kw in keywords:
            if kw in msg.text.lower():
                msg.reply(*ensure_list(reply))
                return True


# 验证入群口令
def valid(msg):
    return group_code in str(msg.text).lower()


# 自动选择未满的群
def get_group():
    groups.sort(key=len, reverse=True)

    for _group in groups:
        if len(_group) < 490:
            return _group
    else:
        logger.warning('群都满啦！')
        return groups[-1]


# 计算每个用户被邀请的次数
invite_counter = Counter()


# 邀请入群
def invite(user):
    joined = list()
    for group in groups:
        if user in group:
            joined.append(group)
    if joined:
        joined_group_names = '\n'.join(map(lambda x: x.name, joined))
        logger.info('{} is already in\n{}'.format(user, joined_group_names))
        user.send('你已加入了\n{}'.format(joined_group_names))
    else:
        if invite_counter.get(user, 0) < 2:
            group = get_group()
            user.send('验证通过 [嘿哈]')
            group.add(user, use_invitation=True)
            invite_counter.update([user])
        else:
            user.send('你的受邀次数已达最大限制 😷')


# 限制频率: 指定周期内超过消息条数，直接回复 "🙊"
def freq_limit(period_secs=15, limit_msgs=5):
    def decorator(func):
        @wraps(func)
        def wrapped(msg):

            if msg.chat in black_list:
                return

            now = datetime.datetime.now()
            period = datetime.timedelta(seconds=period_secs)
            recent_received = 0
            for m in msg.bot.messages[::-1]:
                if m.sender == msg.sender:
                    if now - m.create_time > period:
                        break
                    recent_received += 1

            if recent_received > 8:
                black_list.set(msg.chat, 2 * 3600)
                return '你说得好快，我都累了，休息一下吧'
            elif recent_received > limit_msgs:
                if not isinstance(msg.chat, Group) or msg.is_at:
                    return '🙊'
            return func(msg)

        return wrapped

    return decorator


@dont_raise_response_error
def try_send(chat, msg):
    """尝试发送消息给指定聊天对象"""

    if chat.is_friend:
        chat.send(msg)


def _kick(to_kick, limit_secs=0, msg=None):
    if limit_secs:
        # 加入计时黑名单
        black_list.set(to_kick, limit_secs)

    to_kick.remove()
    start_new_thread(try_send, kwargs=dict(chat=to_kick, msg=msg))

    ret = '@{} 已被成功移出! 😈'.format(to_kick.name)

    if to_kick in kick_votes:
        voters = kick_votes[to_kick][0]
        voters = '\n'.join(map(lambda x: '@{}'.format(x.name), voters))
        ret += '\n\n投票人:\n{}'.format(voters)

    return ret


def remote_kick(msg):
    info_msg = '抱歉，你已被{}移出，接下来的 24 小时内，机器人将对你保持沉默 😷'
    limit_secs = 3600 * 24

    if msg.type == TEXT:
        match = rp_kick.search(msg.text)
        if match:
            name_to_kick = match.group(1)

            try:
                member_to_kick = msg.chat.get(name=name_to_kick)
            except ValueError:
                member_to_kick = msg.chat.get(nickname=name_to_kick)

            # Todo: 有重名时的多个选择

            if member_to_kick in admins:
                logger.error('{} tried to kick {} whom was an admin'.format(
                    msg.member.name, member_to_kick.name))
                return '无法移出管理员 @{} 😷️'.format(member_to_kick.name)

            if from_admin(msg):
                # 管理员: 直接踢出
                return _kick(member_to_kick, limit_secs, info_msg.format('管理员'))
            else:
                # 其他群成员: 投票踢出
                votes, secs_left = kick_votes.vote(voter=msg.member, to_kick=member_to_kick)

                now = time.time()
                voted = 0
                for voters, start in kick_votes.votes.values():
                    if msg.member in voters and now - start < 600:
                        # 10 分钟内尝试投票移出 3 个群员，则认为是恶意用户
                        voted += 1
                        if voted >= 3:
                            _kick(
                                msg.member, limit_secs,
                                '抱歉，你因恶意投票而被移出。接下来的 24 小时内，机器人将对你保持沉默 [悠闲]'
                            )
                            return '移出了恶意投票者 @{} [闪电]'.format(msg.member.name)

                if votes < votes_to_kick:
                    return '正在投票移出 @{}' \
                           '\n当前 {} / {} 票 ({:.0f} 秒有效)' \
                           '\n移出将拉黑 24 小时 😵' \
                           '\n请谨慎投票 🤔'.format(name_to_kick, votes, votes_to_kick, secs_left)
                else:
                    return _kick(member_to_kick, limit_secs, info_msg.format('投票'))


# 判断消息是否为支持回复的消息类型
def supported_msg_type(msg, reply_unsupported=False):
    supported = (TEXT,)
    ignored = (UNKNOWN, NOTICE, NEW_FRIEND)

    fallback_replies = {
        VOICE: '🙉',
        IMAGE: '🙈',
        VIDEO: '🙈',
    }

    if msg.type in supported:
        return True
    elif reply_unsupported and (msg.type not in ignored):
        msg.reply(fallback_replies.get(msg.type, '🐒'))


# 响应好友消息，限制频率
@bot.register(Friend)
@freq_limit()
def exist_friends(msg):
    if msg.chat in black_list:
        return
    elif supported_msg_type(msg, reply_unsupported=True):
        if msg.type == TEXT:
            if valid(msg):
                invite(msg.sender)
            elif str(msg.text).strip().lower() == 'preview':
                if msg.chat in preview_group:
                    return '你已经加入了 preview 群'
                else:
                    msg.reply_file('wxpy-0.5-api-changes.xlsx')
                    msg.reply('''安装 wxpy new-core 分支:
pip3 install -U git+https://github.com/youfou/wxpy.git@new-core

⚠️ 重要说明
部分接口已调整，请见附件表格

⏳ 已知问题
1. 文档仍为滞后状态，会逐步跟进
2. Chat.puid 暂未实现
3. Message.forward() 暂未实现

💬 提交 issue
1. 说明问题的复现步骤 (另附代码效果更佳)
2. 贴出完整的 traceback 信息
3. 为 issue 选择 [new-core] 标签''')

                    preview_group.add(msg.chat, use_invitation=True)
            elif not reply_by_keyword(msg):
                tuling.do_reply(msg)


# 响应好友请求
@bot.register(msg_types=NEW_FRIEND)
def new_friend(msg):
    if msg.card in black_list:
        return
    user = msg.card.accept()
    if valid(msg):
        invite(user)


# 手动加为好友后自动发送消息
@bot.register(Friend, NOTICE)
def manually_added(msg):
    if '现在可以开始聊天了' in msg.text:
        # 对于好友验证信息为 wxpy 的，会等待邀请完成 (并计入 invite_counter)
        # 对于好友验证信息不为 wxpy 的，延迟发送更容易引起注意
        time.sleep(3)
        for group in groups:
            if msg.chat in group:
                break
        else:
            if msg.chat not in invite_counter:
                return '你好呀，{}，还记得咱们的入群口令吗？回复口令即可获取入群邀请。'.format(msg.chat.name)


# 在其他群中回复被 @ 的消息
@bot.register(Group, TEXT)
def reply_other_group(msg):
    if msg.chat not in groups and msg.is_at:
        if supported_msg_type(msg, reply_unsupported=True):
            tuling.do_reply(msg)


# wxpy 群的消息处理
@bot.register([*groups, preview_group], TEXT, except_self=False)
def wxpy_group(msg):
    kick_msg = remote_kick(msg)
    if kick_msg:
        return kick_msg
    elif msg.text.lower().strip() in ('帮助', '说明', '规则', 'help', 'rule', 'rules'):
        return help_info
    elif msg.is_at:
        return 'oops…\n本群禁止使用机器人[撇嘴]\n咱们私聊呗[害羞]'


@bot.register(test_group)
def forward_test_msg(msg):
    if msg.type == TEXT:
        ret = wxpy_group(msg)
        if ret:
            return ret
        elif msg.text == 'text':
            return 'Hello!'
        elif msg.text == 'at':
            return 'Hello @{} !'.format(msg.member.name)


@bot.register((*admins, admin_group), msg_types=TEXT, except_self=False)
def reply_admins(msg):
    """
    响应远程管理员

    内容解析方式优先级：
    1. 若为远程命令，则执行远程命令 (额外定义，一条命令对应一个函数)
    2. 若消息文本以 ! 开头，则作为 shell 命令执行
    3. 尝试作为 Python 代码执行 (可执行大部分 Python 代码)
    4. 若以上不满足或尝试失败，则作为普通聊天内容回复
    """

    try:
        # 上述的 1. 2. 3.
        server_mgmt(msg)
    except ValueError:
        # 上述的 4.
        if isinstance(msg.chat, User):
            return exist_friends(msg)


# 新人入群时激活群数据的更新 (有新消息时才会更新)
@bot.register(groups, NOTICE)
def group_notice(msg):
    admin_group.send('{}:\n{}'.format(msg.chat, msg.text))


def get_logger(level=logging.DEBUG, file='bot.log', mode='a'):
    log_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    log_formatter_lite = logging.Formatter('%(name)s:%(levelname)s:%(message)s')

    _logger = logging.getLogger()

    for hdlr in _logger.handlers:
        _logger.removeHandler(hdlr)

    # 输出到文件
    if file:
        file_hdlr = logging.FileHandler(file, mode)
        file_hdlr.setFormatter(log_formatter)
        _logger.addHandler(file_hdlr)

    # 输出到屏幕
    console_hdlr = logging.StreamHandler()
    console_hdlr.setLevel(logging.WARNING)
    console_hdlr.setFormatter(log_formatter)
    _logger.addHandler(console_hdlr)

    # 输出到远程管理员微信
    wechat_hdlr = WeChatLoggingHandler(admins[0])
    wechat_hdlr.setLevel(logging.WARNING)
    wechat_hdlr.setFormatter(log_formatter_lite)
    _logger.addHandler(wechat_hdlr)

    # 将未捕捉异常也发送到日志中

    def except_hook(*args):
        logger.critical('UNCAUGHT EXCEPTION:', exc_info=args)
        _restart()

    sys.excepthook = except_hook

    for m in 'requests', 'urllib3':
        logging.getLogger(m).setLevel(logging.ERROR)

    _logger.setLevel(level)
    return _logger


logger = get_logger()

send_iter(admin_group, status_text())

bot.join()
