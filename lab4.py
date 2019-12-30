import random
from copy import deepcopy
from uuid import uuid4
import numpy
import matplotlib.pyplot as plt

from file_utils import write_csv_log


class LogRecord:
    TASK_ARRIVED = 'task_arrived'
    TASK_STARTED_PROCESSING = 'task_started_processing'
    TASK_ENDED_PROCESSING = 'task_ended_processing'
    TASK_REJECTED = 'task_rejected'
    TASK_QUEUED = 'task_queued'

    def __init__(self, current_time, event_type, task, queue_length, channels_in_use):
        self.current_time = current_time
        self.event_type = event_type
        self.task = deepcopy(task)
        self.queue_length = queue_length
        self.channels_in_use = channels_in_use

    def __str__(self):
        return f"{self.current_time} :: {self.event_type} {self.task} | Q: {self.queue_length} C: {self.channels_in_use}"


class Task:
    def __init__(self, uid, time_arrived):
        self.uid = uid
        self.time_arrived = time_arrived
        self.time_processing_started = None
        self.time_processing_ends = None
        self.time_entered_queue = None
        self.success = None

    def __str__(self):
        s = f"T{str(self.uid)[:6]} Arrive:{self.time_arrived}"

        if self.time_processing_started is not None:
            s += f" Start:{self.time_processing_started} End:{self.time_processing_ends}"
        if self.success is not None:
            s += f"Success: {self.success}"
        return s


class QueuingSystem:
    def __init__(self, m, n, mu, lambda_, p, time_limit, delta_tau):
        self.m = m  # мест в очереди
        self.n = n  # касс
        self.mu = mu  # интенсивность потока обслуживания
        self.lambda_ = lambda_  # интенсивность потока входа
        self.p = p  # вероятность того, что заяка будет обслужена
        self.queue = []
        self.channels = []
        self.current_time = 0
        self.time_limit = time_limit
        self.delta_tau = delta_tau
        self.pending_task = None
        self.logs = []
        self.arrived_tasks_N = 0
        self.processed_tasks_N = 0
        self.rejected_tasks_N = 0
        self.successfully_processed_tasks_N = 0

    def get_current_time(self):
        return round(self.current_time,3)

    def generate_single_task_service_time(self):
        return random.expovariate(self.mu)

    def generate_single_task_arrive_time(self):
        return random.expovariate(self.lambda_)

    def generate_task_service_success(self):
        return numpy.random.choice(2, 1, replace=False, p=[1-self.p, self.p])[0]

    def try_insert_task(self, task):

        if self.try_process_task(task):
            return True
        if len(self.queue) < self.m:
            task.time_entered_queue = self.current_time
            self.queue.append(task)
            return True
        self.log(LogRecord.TASK_REJECTED, task)
        return False

    def try_process_task(self, task):
        if len(self.channels) < self.n:
            task.time_processing_started = self.get_current_time()
            task.time_processing_ends = self.get_current_time() + \
                self.generate_single_task_service_time()
            self.channels.append(task)
            self.log(LogRecord.TASK_STARTED_PROCESSING, task)
            return True
        return False

    def log(self, event, task):
        #print(self.current_time, '::', event, task, *messages)
        if event == LogRecord.TASK_ARRIVED:
            self.arrived_tasks_N += 1
        elif event == LogRecord.TASK_ENDED_PROCESSING:
            self.processed_tasks_N += 1
            if task.success:
                self.successfully_processed_tasks_N += 1
        elif event == LogRecord.TASK_REJECTED:
            self.rejected_tasks_N += 1
        
        self.logs.append(LogRecord(self.get_current_time(), event,
                                   task, len(self.queue), len(self.channels)))

    def step(self):
        # проверить, пришло ли время для появления новой заявки
        if self.pending_task and self.pending_task.time_arrived < self.get_current_time():
            self.log(LogRecord.TASK_ARRIVED, self.pending_task)
            self.try_insert_task(self.pending_task)
            self.pending_task = Task(
                uuid4(), self.generate_single_task_service_time())
        elif self.pending_task is None:
            self.pending_task = Task(
                uuid4(), self.generate_single_task_service_time())

        # сделать шаг обработки
        still_processing_tasks = []
        processed_tasks = []
        for task in self.channels:
            if task.time_processing_ends <= self.get_current_time():
                task.success = self.generate_task_service_success()
                processed_tasks.append(task)
            else:
                still_processing_tasks.append(task)
        self.channels = still_processing_tasks

        if len(self.queue) > 0:
            if self.try_process_task(self.queue[0]):
                self.queue.pop(0)

        # попробовать снова отправить в обработку неполучившуюся задачу
        for task in processed_tasks:
            self.log(LogRecord.TASK_ENDED_PROCESSING, task)
            if not task.success:
                self.try_insert_task(task)

    def run(self):
        while self.get_current_time() < self.time_limit:
            self.step()
            self.current_time += self.delta_tau


if __name__ == "__main__":
    T = 100 # верхняя граница времени
    M = 10 # количество мест в очереди
    N = 5 # число каналов обслуживания
    lambda_ = 8 # интенсивность входного потока
    mu = 4 # інтенсивность потока обслуживания 
    p = 0.99 # вероятность успешного обслуживания заявки в канале
    delta_T = 0.02 # шаг времени
    qs = QueuingSystem(M, N, mu, lambda_, p, T, delta_T)
    qs.run()

    steps = []
    queue_states = []
    channels_states = []
    channels_states_by_channels = [0]*(N+1)
    last = qs.logs[0]
    for record in qs.logs:
        if last.current_time != record:
            steps.append(last.current_time)
            queue_states.append(last.queue_length)
            channels_states_by_channels[last.channels_in_use] += 1
            channels_states.append(last.channels_in_use)
        last = record

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey='row')
    axes[0].plot(steps, channels_states, label="channels busy")
    axes[0].set_title('Channels busy')  
    axes[1].plot(steps, queue_states, 'orange', label="tasks in queue")
    axes[1].set_title('Queue size')
    plt.tight_layout()
    plt.show()

    print('Вероятность отказа = ', qs.rejected_tasks_N / qs.arrived_tasks_N)
    print('Вероятность обслуживания = ', qs.processed_tasks_N/ qs.arrived_tasks_N)
    print('Пропускная способность = ',qs.processed_tasks_N / T, 'заявок в единицу времени')
    print('Среднее число занятых каналов = ', sum(channels_states)/len(channels_states))
    for i, c in enumerate(channels_states_by_channels):
        print(f'Р({i} каналов занято) =', c/len(channels_states))

    ro = lambda_/mu
    

    write_csv_log(qs.logs)
    