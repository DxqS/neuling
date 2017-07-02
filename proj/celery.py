# coding:utf-8
'''
Created on 2017/7/2.

@author: Dxq
'''
from __future__ import absolute_import
from celery import Celery
from celery.schedules import crontab
import config

redis_url = "redis://:{password}@{host}:{port}/{db}".format(**config.srv['redis'])
# rabbit_url = "amqp://{user}:{pwd}@{host}:{port}/{vhost}".format(**config.srv['rabbit'])

app = Celery('proj', broker=redis_url, include=['proj.tasks'])

CELERYBEAT_SCHEDULE = {

}
# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_RESULT_EXPIRES=3600,
    CELERY_TASK_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],  # Ignore other content
    CELERY_RESULT_SERIALIZER='json',
    CELERY_TIMEZONE='Asia/Shanghai',
    CELERY_ENABLE_UTC=True,
    CELERYBEAT_SCHEDULE=CELERYBEAT_SCHEDULE
)

if __name__ == '__main__':
    app.start()

    # celery -A test worker --loglevel=info
    # celery -A proj worker -l info
    # celery -A proj beat
    # celery multi start w1 -A proj -l info
    # celery multi restart w1 -A proj -l info
    # celery multi stop w1 -A proj -l info
    # celery multi stopwait w1 -A proj -l info
